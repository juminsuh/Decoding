# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Hub utilities: utilities related to download and cache models
"""
import copy
import fnmatch
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import warnings
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZipFile, is_zipfile
import urllib.parse #### 수정


import requests
from filelock import FileLock
from huggingface_hub import HfFolder, Repository, create_repo, list_repo_files, whoami
from requests.exceptions import HTTPError
from requests.models import Response
from transformers.utils.logging import tqdm

from . import __version__, logging
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _tf_version,
    _torch_version,
    is_tf_available,
    is_torch_available,
    is_training_run_on_sagemaker,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
# New default cache, shared with the Datasets library
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "transformers")

# Onetime move from the old location to the new one if no ENV variable has been set.
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(default_cache_path)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    logger.warning(
        "In Transformers v4.0.0, the default path to cache downloaded models changed from"
        " '~/.cache/torch/transformers' to '~/.cache/huggingface/transformers'. Since you don't seem to have"
        " overridden and '~/.cache/torch/transformers' is a directory that exists, we're moving it to"
        " '~/.cache/huggingface/transformers' to avoid redownloading models you have already in the cache. You should"
        " only see this message once."
    )
    shutil.move(old_default_cache_path, default_cache_path)

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://moon-staging.huggingface.co" if _staging_mode else "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    warnings.warn(
        "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
        "Transformers v5. Use `HF_ENDPOINT` instead.",
        FutureWarning,
    )
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None)
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if mirror:
        if mirror in ["tuna", "bfsu"]:
            raise ValueError("The Tuna and BFSU mirrors are no longer available. Try removing the mirror argument.")
        legacy_format = "/" not in model_id
        if legacy_format:
            return f"{mirror}/{model_id}-{filename}"
        else:
            return f"{mirror}/{model_id}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be `None`) stored for *filename*. Raise `EnvironmentError` if *filename* or its
    stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError(f"file {cache_path} not found")

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError(f"file {meta_path} not found")

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape `(model_url,
    etag, size_MB)`. Filenames in `cache_dir` are use to get the metadata for each model, only urls ending with *.bin*
    are added.

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape `(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(f"Archive format of {output_path} could not be identified")

        return output_path_extracted

    return output_path


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", 0),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", 0),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


class RepositoryNotFoundError(HTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or with a private repo name the user does
    not have access to.
    """


class EntryNotFoundError(HTTPError):
    """Raised when trying to access a hf.co URL with a valid repository and revision but an invalid filename."""


class RevisionNotFoundError(HTTPError):
    """Raised when trying to access a hf.co URL with a valid repository but an invalid revision."""


def _raise_for_status(response: Response):
    """
    Internal version of `request.raise_for_status()` that will refine a potential HTTPError.
    """
    if "X-Error-Code" in response.headers:
        error_code = response.headers["X-Error-Code"]
        if error_code == "RepoNotFound":
            raise RepositoryNotFoundError(f"404 Client Error: Repository Not Found for url: {response.url}")
        elif error_code == "EntryNotFound":
            raise EntryNotFoundError(f"404 Client Error: Entry Not Found for url: {response.url}")
        elif error_code == "RevisionNotFound":
            raise RevisionNotFoundError(f"404 Client Error: Revision Not Found for url: {response.url}")

    if response.status_code == 401:
        # The repo was not found and the user is not Authenticated
        raise RepositoryNotFoundError(
            f"401 Client Error: Repository not found for url: {response.url}. "
            "If the repo is private, make sure you are authenticated."
        )

    response.raise_for_status()

########### 수정 #############
# def _normalize_etag(etag: str) -> str:
#     """ETag의 따옴표/weak 프리픽스를 제거하여 캐시 키로 쓰기 좋게 만든다."""
#     if not etag:
#         return etag
#     e = etag.strip()
#     if e.startswith("W/"):
#         e = e[2:]
#     if len(e) >= 2 and e[0] == '"' and e[-1] == '"':
#         e = e[1:-1]
#     return e

# ETAG_HEADER_CANDIDATES = ("X-Linked-Etag", "ETag")
# from urllib.parse import urljoin

# def _head_follow_final(
#     url: str,
#     headers: Dict[str, str],
#     proxies=None,
#     timeout: int = 10,
#     max_redirects: int = 5,
# ):
#     """
#     HEAD를 allow_redirects=False로 수행하며 최종 목적지 URL과 etag를 얻는다.
#     중간에 상대 Location이 오면 절대 URL로 보정한다.
#     """
#     cur = url
#     for _ in range(max_redirects):
#         r = requests.head(
#             cur, headers=headers, allow_redirects=False, proxies=proxies, timeout=timeout
#         )
#         _raise_for_status(r)

#         # 리다이렉트면 Location을 절대 URL로 변환하고 반복
#         if 300 <= r.status_code <= 399:
#             loc = r.headers.get("Location")
#             if not loc:
#                 raise OSError("Redirect without Location header")
#             cur = urljoin(cur, loc)
#             continue

#         # 최종 응답(200대)
#         etag = None
#         for k in ETAG_HEADER_CANDIDATES:
#             if r.headers.get(k):
#                 etag = r.headers[k]
#                 break
#         if not etag:
#             raise OSError("No ETag/X-Linked-Etag on final resource")
#         return cur, _normalize_etag(etag)

#     raise OSError("Too many redirects while resolving HEAD")

# def http_get(
#     url: str,
#     temp_file: BinaryIO,
#     proxies=None,
#     resume_size=0,
#     headers: Optional[Dict[str, str]] = None,
#     expected_etag: Optional[str] = None,
#     max_redirects: int = 5,
# ):
#     """
#     최종 URL을 기준으로 파일을 내려받는다. 리다이렉트는 수동으로 따라가며,
#     가능하면 ETag 일치도 검증한다.
#     """
#     # print(f"[DEBUG] HTTP GET CALLED WITH URL: {url}")

#     # 절대 URL 가드
#     if not urlparse(url).scheme:
#         raise OSError(f"GET url is not absolute: {url}")

#     headers = copy.deepcopy(headers) if headers else {}
#     if resume_size > 0:
#         headers["Range"] = f"bytes={resume_size}-"

#     # 수동 리다이렉트 추적(최대 max_redirects회)
#     cur = url
#     redirs = 0
#     while True:
#         r = requests.get(
#             cur, stream=True, proxies=proxies, headers=headers, allow_redirects=False
#         )
#         if 300 <= r.status_code <= 399 and redirs < max_redirects:
#             loc = r.headers.get("Location")
#             if not loc:
#                 r.close()
#                 raise OSError("GET redirect without Location header")
#             cur = urljoin(cur, loc)
#             redirs += 1
#             r.close()
#             continue
#         break

#     _raise_for_status(r)

#     # ETag 검증(서버가 제공하면)
#     got_etag = None
#     for k in ETAG_HEADER_CANDIDATES:
#         if r.headers.get(k):
#             got_etag = _normalize_etag(r.headers[k])
#             break
#     if expected_etag and got_etag and (expected_etag != got_etag):
#         r.close()
#         raise OSError(f"ETag mismatch: expected {expected_etag}, got {got_etag}")

#     content_length = r.headers.get("Content-Length")
#     total = resume_size + int(content_length) if content_length is not None else None

#     progress = tqdm(
#         unit="B",
#         unit_scale=True,
#         unit_divisor=1024,
#         total=total,
#         initial=resume_size,
#         desc="Downloading",
#     )
#     # 큰 청크로 빠르게
#     for chunk in r.iter_content(chunk_size=1024 * 256):
#         if chunk:  # keep-alive 차단
#             progress.update(len(chunk))
#             temp_file.write(chunk)
#     progress.close()
#     r.close()

# def get_from_cache(
#     url: str,
#     cache_dir=None,
#     force_download=False,
#     proxies=None,
#     etag_timeout=10,
#     resume_download=False,
#     user_agent: Union[Dict, str, None] = None,
#     use_auth_token: Union[bool, str, None] = None,
#     local_files_only=False,
# ) -> Optional[str]:
#     """
#     URL에 해당하는 파일을 로컬 캐시에서 찾고, 없으면 다운로드해서 캐시에 넣은 뒤 경로를 반환.
#     여기서는 '최종 HEAD의 ETag'를 캐시 키로 사용하고 GET도 동일 최종 URL로 호출한다.
#     """
#     if cache_dir is None:
#         cache_dir = TRANSFORMERS_CACHE
#     if isinstance(cache_dir, Path):
#         cache_dir = str(cache_dir)

#     os.makedirs(cache_dir, exist_ok=True)

#     headers = {"user-agent": http_user_agent(user_agent)}
#     if isinstance(use_auth_token, str):
#         headers["authorization"] = f"Bearer {use_auth_token}"
#     elif use_auth_token:
#         token = HfFolder.get_token()
#         if token is None:
#             raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
#         headers["authorization"] = f"Bearer {token}"

#     # 오프라인 모드면 네트워크 접근 금지
#     if local_files_only:
#         # local_files_only 경로 탐색은 당신의 기존 캐시 스킴에 맞게 구현하세요.
#         # 여기서는 단순히 캐시 폴더 내 파일 탐색을 생략합니다.
#         pass

#     # 1) 최종 HEAD 따라가며 final_url / normalized etag 획득
#     url_to_download = url
#     etag = None
#     if not local_files_only:
#         # print(f"[DEBUG] url_to_download(before HEAD): {url_to_download}")
#         final_url, final_etag = _head_follow_final(
#             url, headers=headers, proxies=proxies, timeout=etag_timeout, max_redirects=5
#         )
#         url_to_download = final_url
#         etag = final_etag
#         # print(f"[DEBUG] FINAL HEAD url: {url_to_download}")
#         # print(f"[DEBUG] FINAL HEAD etag(normalized): {etag}")

#     # 2) 캐시 경로 결정 (파일명 + etag)
#     #    참고: 실제 HF Hub는 blobs/snapshots 구조를 쓰지만, 여기서는 간단 캐시를 예시로 보여줌
#     parsed = urlparse(url_to_download)
#     filename = os.path.basename(parsed.path) or "unknown"
#     safe_etag = etag or "noetag"
#     cache_basename = f"{filename}.{safe_etag}"
#     cache_path = os.path.join(cache_dir, cache_basename)

#     # 3) 캐시 적중 처리
#     if not force_download and os.path.exists(cache_path):
#         # print(f"[DEBUG] Cache hit: {cache_path}")
#         return cache_path

#     if local_files_only:
#         # 오프라인인데 캐시에 없으면 실패
#         raise FileNotFoundError(f"File not found in cache (offline mode): {cache_path}")

#     # 4) 다운로드 (resume 옵션 포함)
#     tmp_suffix = ".incomplete"
#     tmp_path = cache_path + tmp_suffix
#     resume_size = 0
#     if resume_download and os.path.exists(tmp_path):
#         resume_size = os.path.getsize(tmp_path)

#     with open(tmp_path, "ab") as tmp:
#         http_get(
#             url_to_download,
#             tmp,
#             proxies=proxies,
#             resume_size=resume_size,
#             headers=headers,
#             expected_etag=etag,
#         )

#     # 5) 원자적 이동
#     os.replace(tmp_path, cache_path)
#     # print(f"[DEBUG] Cached at: {cache_path}")
#     return cache_path
###########################

def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, headers: Optional[Dict[str, str]] = None):
    """
    Download remote file. Do not gobble up errors.
    """
    print(f"[DEBUG] HTTP GET CALLED WITH URL: {url}")
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers["Range"] = f"bytes={resume_size}-"
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    _raise_for_status(r)
    content_length = r.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
    # and can be set using `utils.logging.enable/disable_progress_bar()`
    progress = tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=total,
        initial=resume_size,
        desc="Downloading",
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            print(f"[DEBUG] url_to_download: {url_to_download}")
            # print(f"[DEBUG] url: {url}")
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            _raise_for_status(r)
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399: #### 수정
                redirected = r.headers["Location"]
                print(f"[DEBUG] REDIRECT Location Header: {redirected}")
                
                # 상대경로일 경우 절대 URL로 보정
                if redirected.startswith("/"):
                    from urllib.parse import urljoin, urlparse

                    base_url = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
                    url_to_download = urljoin(base_url, redirected)
                    # url_to_download = f"https://{redirected}"
                    print(f"[DEBUG] Converted relative redirect to absolute: {url_to_download}")
                else:
                    url_to_download = redirected
                # url_to_download = r.headers["Location"]
                # print(f"[DEBUG] REDIRECTED Location Header: {url_to_download}")
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ProxyError,
            RepositoryNotFoundError,
            EntryNotFoundError,
            RevisionNotFoundError,
        ):
            # Actually raise for those subclasses of ConnectionError
            # Also raise the custom errors coming from a non existing repo/branch/file as they are caught later on.
            raise
        except (HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")

            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)
    
    # print(f"[DEBUG] FINAL CACHED PATH: {cache_path}")

    return cache_path


def get_file_from_repo(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.

    <Tip>

    Passing `use_auth_token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("xlm-roberta-base", "tokenizer_config.json")
    ```"""
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    path_or_repo = str(path_or_repo)
    if os.path.isdir(path_or_repo):
        resolved_file = os.path.join(path_or_repo, filename)
        return resolved_file if os.path.isfile(resolved_file) else None
    else:
        resolved_file = hf_bucket_url(path_or_repo, filename=filename, revision=revision, mirror=None)

    try:
        # Load from URL or cache if already cached
        resolved_file = cached_path(
            resolved_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
        )

    except RepositoryNotFoundError:
        raise EnvironmentError(
            f"{path_or_repo} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to "
            "pass a token having permission to this repo with `use_auth_token` or log in with "
            "`huggingface-cli login` and pass `use_auth_token=True`."
        )
    except RevisionNotFoundError:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo}' for available revisions."
        )
    except EnvironmentError:
        # The repo and revision exist, but the file does not or there was a connection error fetching it.
        return None

    return resolved_file


def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    mirror: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
):
    """
    Checks if a repo contains a given file wihtout downloading it. Works for remote repos and local folders.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    url = hf_bucket_url(path_or_repo, filename=filename, revision=revision, mirror=mirror)

    headers = {"user-agent": http_user_agent()}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = f"Bearer {token}"

    r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
    try:
        _raise_for_status(r)
        return True
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'.")
    except RevisionNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        )
    except requests.HTTPError:
        # We return false for EntryNotFoundError (logical) as well as any connection error.
        return False


def get_list_of_files(
    path_or_repo: Union[str, os.PathLike],
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
) -> List[str]:
    """
    Gets the list of files inside `path_or_repo`.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            Can be either the id of a repo on huggingface.co or a path to a *directory*.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether or not to only rely on local files and not to attempt to download any files.

    <Tip warning={true}>

    This API is not optimized, so calling it a lot may result in connection errors.

    </Tip>

    Returns:
        `List[str]`: The list of files available in `path_or_repo`.
    """
    path_or_repo = str(path_or_repo)
    # If path_or_repo is a folder, we just return what is inside (subdirectories included).
    if os.path.isdir(path_or_repo):
        list_of_files = []
        for path, dir_names, file_names in os.walk(path_or_repo):
            list_of_files.extend([os.path.join(path, f) for f in file_names])
        return list_of_files

    # Can't grab the files if we are on offline mode.
    if is_offline_mode() or local_files_only:
        return []

    # Otherwise we grab the token and use the list_repo_files method.
    if isinstance(use_auth_token, str):
        token = use_auth_token
    elif use_auth_token is True:
        token = HfFolder.get_token()
    else:
        token = None

    try:
        return list_repo_files(path_or_repo, revision=revision, token=token)
    except HTTPError as e:
        raise ValueError(
            f"{path_or_repo} is not a local path or a model identifier on the model Hub. Did you make a typo?"
        ) from e


def is_local_clone(repo_path, repo_url):
    """
    Checks if the folder in `repo_path` is a local clone of `repo_url`.
    """
    # First double-check that `repo_path` is a git repo
    if not os.path.exists(os.path.join(repo_path, ".git")):
        return False
    test_git = subprocess.run("git branch".split(), cwd=repo_path)
    if test_git.returncode != 0:
        return False

    # Then look at its remotes
    remotes = subprocess.run(
        "git remote -v".split(),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=repo_path,
    ).stdout

    return repo_url in remotes.split()


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def push_to_hub(
        self,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        use_temp_dir: bool = False,
        commit_message: Optional[str] = None,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        **model_card_kwargs
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub while synchronizing a local clone of the repo in
        `repo_path_or_name`.

        Parameters:
            repo_path_or_name (`str`, *optional*):
                Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case
                the repository will have the name of that local folder). If not specified, will default to the name
                given by `repo_url` and a local directory with that name will be created.
            repo_url (`str`, *optional*):
                Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
                repository will be created in your namespace (unless you specify an `organization`) with `repo_name`.
            use_temp_dir (`bool`, *optional*, defaults to `False`):
                Whether or not to clone the distant repo in a temporary directory or in `repo_path_or_name` inside the
                current working directory. This will slow things down if you are making changes in an existing repo
                since you will need to clone the repo before every push.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"add {object}"`.
            organization (`str`, *optional*):
                Organization in which you want to push your {object} (you must be a member of this organization).
            private (`bool`, *optional*):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`). Will default to `True` if
                `repo_url` is not specified.


        Returns:
            `str`: The url of the commit of your {object} in the given repository.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert" and have a local clone in the
        # *my-finetuned-bert* folder.
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to your namespace with the name "my-finetuned-bert" with no local clone.
        {object}.push_to_hub("my-finetuned-bert", use_temp_dir=True)

        # Push the {object} to an organization with the name "my-finetuned-bert" and have a local clone in the
        # *my-finetuned-bert* folder.
        {object}.push_to_hub("my-finetuned-bert", organization="huggingface")

        # Make a change to an existing repo that has been cloned locally in *my-finetuned-bert*.
        {object}.push_to_hub("my-finetuned-bert", repo_url="https://huggingface.co/sgugger/my-finetuned-bert")
        ```
        """
        if use_temp_dir:
            # Make sure we use the right `repo_name` for the `repo_url` before replacing it.
            if repo_url is None:
                if use_auth_token is None:
                    use_auth_token = True
                repo_name = Path(repo_path_or_name).name
                repo_url = self._get_repo_url_from_name(
                    repo_name, organization=organization, private=private, use_auth_token=use_auth_token
                )
            repo_path_or_name = tempfile.mkdtemp()

        # Create or clone the repo. If the repo is already cloned, this just retrieves the path to the repo.
        repo = self._create_or_get_repo(
            repo_path_or_name=repo_path_or_name,
            repo_url=repo_url,
            organization=organization,
            private=private,
            use_auth_token=use_auth_token,
        )
        # Save the files in the cloned repo
        self.save_pretrained(repo_path_or_name)
        if hasattr(self, "history") and hasattr(self, "create_model_card"):
            # This is a Keras model and we might be able to fish out its History and make a model card out of it
            base_model_card_args = {
                "output_dir": repo_path_or_name,
                "model_name": Path(repo_path_or_name).name,
            }
            base_model_card_args.update(model_card_kwargs)
            self.create_model_card(**base_model_card_args)
        # Commit and push!
        url = self._push_to_hub(repo, commit_message=commit_message)

        # Clean up! Clean up! Everybody everywhere!
        if use_temp_dir:
            shutil.rmtree(repo_path_or_name)

        return url

    @staticmethod
    def _get_repo_url_from_name(
        repo_name: str,
        organization: Optional[str] = None,
        private: bool = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        elif use_auth_token:
            token = HfFolder.get_token()
            if token is None:
                raise ValueError(
                    "You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and "
                    "entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own "
                    "token as the `use_auth_token` argument."
                )
        else:
            token = None

        # Special provision for the test endpoint (CI)
        return create_repo(
            token,
            repo_name,
            organization=organization,
            private=private,
            repo_type=None,
            exist_ok=True,
        )

    @classmethod
    def _create_or_get_repo(
        cls,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
        private: bool = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> Repository:
        if repo_path_or_name is None and repo_url is None:
            raise ValueError("You need to specify a `repo_path_or_name` or a `repo_url`.")

        if use_auth_token is None and repo_url is None:
            use_auth_token = True

        if repo_path_or_name is None:
            repo_path_or_name = repo_url.split("/")[-1]

        if repo_url is None and not os.path.exists(repo_path_or_name):
            repo_name = Path(repo_path_or_name).name
            repo_url = cls._get_repo_url_from_name(
                repo_name, organization=organization, private=private, use_auth_token=use_auth_token
            )

        # Create a working directory if it does not exist.
        if not os.path.exists(repo_path_or_name):
            os.makedirs(repo_path_or_name)

        repo = Repository(repo_path_or_name, clone_from=repo_url, use_auth_token=use_auth_token)
        repo.git_pull()
        return repo

    @classmethod
    def _push_to_hub(cls, repo: Repository, commit_message: Optional[str] = None) -> str:
        if commit_message is None:
            if "Tokenizer" in cls.__name__:
                commit_message = "add tokenizer"
            elif "Config" in cls.__name__:
                commit_message = "add config"
            else:
                commit_message = "add model"

        return repo.push_to_hub(commit_message=commit_message)


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def send_example_telemetry(example_name, *example_args, framework="pytorch"):
    """
    Sends telemetry that helps tracking the examples use.

    Args:
        example_name (`str`): The name of the example.
        *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script. This function will only
            try to extract the model and dataset name from those. Nothing else is tracked.
        framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    """
    if is_offline_mode():
        return

    data = {"example": example_name, "framework": framework}
    for args in example_args:
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and v is not None}
        if "model_name_or_path" in args_as_dict:
            model_name = args_as_dict["model_name_or_path"]
            # Filter out local paths
            if not os.path.isdir(model_name):
                data["model_name"] = args_as_dict["model_name_or_path"]
        if "dataset_name" in args_as_dict:
            data["dataset_name"] = args_as_dict["dataset_name"]
        elif "task_name" in args_as_dict:
            # Extract script name from the example_name
            script_name = example_name.replace("tf_", "").replace("flax_", "").replace("run_", "")
            script_name = script_name.replace("_no_trainer", "")
            data["dataset_name"] = f"{script_name}-{args_as_dict['task_name']}"

    headers = {"user-agent": http_user_agent(data)}
    try:
        r = requests.head(HUGGINGFACE_CO_EXAMPLES_TELEMETRY, headers=headers)
        r.raise_for_status()
    except Exception:
        # We don't want to error in case of connection errors of any kind.
        pass
