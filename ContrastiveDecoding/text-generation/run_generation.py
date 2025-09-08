#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm 
import json 
import config
from accelerate import dispatch_model
from datasets import Dataset
import datasets

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer), # gpt-2
    "opt": (OPTForCausalLM, GPT2Tokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer)
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#

# get_len = 31 
# opt-ready: past key values (캐시같은 개념)이 존재할 때 prefix의 last token만을 이용해서 student_lm의 context window를 제한
def ignore_prefix_opt_prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
    if past is None:
        input_ids = input_ids[:, -1:]
    else:
        # print(past[0][0].shape) 
        genlen = past[0][0].shape[2] 
        input_ids = input_ids[:, -(genlen + 1):]
    # print(input_ids.shape) 

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    input_ids = input_ids[:, -1:]
    # print(attention_mask.shape, input_ids.shape, 'ignore_prefix') 
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }

# gpt-ready: past key values (캐시같은 개념)이 존재할 때 prefix의 last token만을 이용해서 student_lm의 context window를 제한
def ignore_prefix_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
            
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def out_file(args, generation_lst):
    # dataset:contrastive_decoding 여부 (student/no):디코딩 종류 (no/contrastive_search_baseline/top-k/nucleus/typical/eta)
    output_path = os.path.join("/mnt/aix7101/minsuh/decoding_results", f"{args.model_type}_{args.prompt_file}_{args.contrastive_decoding}_{args.do_sample}_output.jsonl")
    with open(output_path, 'w') as f:
        for kk in generation_lst:
            print(json.dumps(kk), file=f) 
    print(f'written to {output_path}')
    return 

def format_out(generated_text, prompt, generated_tokens, gold_ref=None):
    output = {
                'ended'      : False,
                'tokens'     : generated_tokens,
                'prompt'     : prompt,
                'gen_text'   : generated_text, 
                'len'        : 0,
                'nll4tok'    : [],
                'ppl4tok'    : [],
                'ppl'        : 0,
                'gold_ref'   : gold_ref, 
            } 
            
    return output 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--student_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--revision",
        default='checkpoint-200000',
        type=str,
    )
    parser.add_argument("--contrastive_decoding", type=str, default="student")
    parser.add_argument("--contrastive_prompt", type=str, default="I love repetitive text! Here is my writing:")
    parser.add_argument("--st_coef", type=float, default=0.5)

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--do_sample", type=str, default="no")
    parser.add_argument("--outfile", type=str, default="outfile.jsonl")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--num_beam", type=int, default=5)

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--min_prob", type=float, default=0.0)

    parser.add_argument("--student_min_prob", type=float, default=0.0)
    parser.add_argument("--student_temperature", type=float, default=1.0)
    parser.add_argument("--use_cap_student", type=str, default='no')
    parser.add_argument("--ignore_prefix", type=str, default='yes') # IMPORTANT
    parser.add_argument("--use_switch", type=str, default='no')
    

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available") # default: False
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    ) # default: False -> use 32-bit
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    return args 

def main(args):
    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        # gpt2_model_class, gpt2_tokenizer_class =  MODEL_CLASSES['gpt2']
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    if args.do_sample == 'contrastive_search_baseline': 
        if 'gpt' in  args.model_name_or_path or "Llama" in args.model_name_or_path:
            from simctg.simctggpt import SimCTGGPT
            model_name = args.model_name_or_path
            model = SimCTGGPT(model_name)
            print("\n🤖 Simctg model is loaded.\n") 
            model.eval()
            tokenizer = model.tokenizer
            eos_token_id = tokenizer.eos_token_id
            model.to(args.device)
        elif 'opt' in args.model_name_or_path:
            from simctg.simctgopt import SimCTGOPT
            model_name = args.model_name_or_path
            print("\n🤖 Simctg model is loaded.\n") 
            model = SimCTGOPT(model_name)
            tokenizer = model.tokenizer
            model.eval()
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
            dispatch_model(model.model, device_map=config.device_map)
            
        else:
            raise NotImplemented
    else:
        print(model_class)
        if args.model_type == "gpt2":
            model = model_class.from_pretrained(args.model_name_or_path) # expert 모델 로드
            model.to(args.device)
        elif args.model_type == "opt":
            model = model_class.from_pretrained(args.model_name_or_path, device_map=config.device_map)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        print(f"🤖 Teacher model is loaded.\n")
        
    if args.fp16:
        model.half()
        
    if args.contrastive_decoding == 'student': # contrastive decoding인 경우에만 student 존재
        assert args.student_name_or_path is not None
        if args.model_type == "gpt2":
            student_lm = AutoModelForCausalLM.from_pretrained(args.student_name_or_path) # load student model 
            student_lm.to(args.device)
        elif args.model_type == "opt":
            student_lm = AutoModelForCausalLM.from_pretrained(args.student_name_or_path, device_map="auto") # load student model 
        print(f"\n🤖 Student model is loaded.\n")
        if args.fp16:
            student_lm.half()
        if args.ignore_prefix == 'yes':
            if 'gpt' in args.model_name_or_path:
                student_lm.prepare_inputs_for_generation = ignore_prefix_prepare_inputs_for_generation # student_lm에게는 prefix의 last token만 보여줌
            elif 'opt' in args.model_name_or_path:
                student_lm.prepare_inputs_for_generation = ignore_prefix_opt_prepare_inputs_for_generation
    else:
        student_lm = None 

    if args.do_sample != 'contrastive_search_baseline':
        args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    if not args.prompt_file: # 직접 prompt 입력
        prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
        prompt_lst = [prompt_text]
        ref_lst = [(0, None)] 
    elif args.prompt_file == 'wikitext' or args.prompt_file == "wp" or args.prompt_file == 'wikinews': # type of dataset
        # load wikitext. 
        from datasets import load_dataset, concatenate_datasets
        if args.prompt_file == 'wikitext':
            datasets_val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
            datasets_test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
            dataset = concatenate_datasets([datasets_val, datasets_test])
            print(dataset)
        elif args.prompt_file == "wp":
            # datasets_val = load_dataset("euclaise/writingprompts", split="validation")
            dataset = load_dataset("euclaise/writingprompts", split="test")
            # datasets = concatenate_datasets([datasets_val, datasets_test])
            print(dataset)
        elif args.prompt_file == 'wikinews':
            # lang=="en"인 것만 남기기
            data_path = '../data/multilingual_wikinews.jsonl'
            save_path = "../data/wikinews_en"
            
            def save_dataset():
                if not os.path.exists(save_path):
                    print("made newly!")
                    texts = []
                    with open(data_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)  
                            if obj.get("lang") == "en":
                                text = obj.get("text", [])
                                texts.append(text[0])

                    dataset = Dataset.from_dict({"text": texts})
                    dataset.save_to_disk(save_path)
                else:
                    print("already exists!")
                    print(f"save_path: {save_path}")
                return save_path

            dataset = datasets.load_from_disk(save_dataset())
            print(dataset)

        column_names = dataset.column_names # wikitext: ['text'], wp: ['prompt', 'story'], wikinews: ['text']
        text_column_name = "text" if "text" in column_names else column_names[0] # wikitext: 'text', wp: 'prompt', wikinews: 'text'
        print(f"text_column_name: {text_column_name}")
        
        def tokenize_function(examples):
            examples[text_column_name] = [x.replace(' <newline>', '\n') for x in examples[text_column_name]]
            if args.prompt_file == "wp":                
                examples["story"] = [x.replace(' <newline>', '\n') for x in examples["story"]]
            examples[text_column_name] = [tokenizer.bos_token + x for x in examples[text_column_name] if len(x) > 0]

            result_dict = tokenizer(examples[text_column_name], add_special_tokens=False) 
            if args.prompt_file == "wikitext": 
                input_ids_lst = [x[:32] for x in result_dict['input_ids'] if len(x) >= 160] # first 32 words as prompt 
                gold_lst = [x for x in result_dict['input_ids'] if len(x) >= 160] # add to golden list if length >= 160
            elif args.prompt_file == "wikinews":
                input_ids_lst = [x[:32] for x in result_dict['input_ids']] # first 32 words as prompt 
                gold_lst = [x for x in result_dict['input_ids']]   
            elif args.prompt_file == "wp":
                story_dict = tokenizer(examples["story"], add_special_tokens=False, truncation=True)
                input_ids_lst = [result_dict['input_ids'][i] for i in range(len(result_dict['input_ids']))] # 데이터셋에 있는 prompt 그대로 사용
                gold_lst = [story_dict['input_ids'][i] for i in range(len(story_dict['input_ids']))]
            result_dict2 = {'input_ids':input_ids_lst, 'gold':gold_lst}
            return result_dict2

        tokenized_datasets = dataset.map( # apply tokenize_function to dataset 
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
        )
        print(tokenized_datasets)
        
        if args.prompt_file == 'wikitext' or args.prompt_file == "wikinews" or args.prompt_file == "wp":
            prompt_ids = tokenized_datasets[:2000]['input_ids'] 
            ref_lst = tokenized_datasets[:2000]['gold'] 
            ref_lst = tokenizer.batch_decode(ref_lst)
            ref_lst = [(0, x) for x in ref_lst]

        prompt_lst = tokenizer.batch_decode(prompt_ids)
        print(f"length of prompt list: {len(prompt_lst)}") # wikitext: 1314, wp: 2000, wikinews: 2000

    generation_lst = []
    
    # decoding
    for iidx, prompt_text in tqdm(enumerate(prompt_lst[:2000]), total=min(2000, len(prompt_lst)), desc="😊 Processing Prompt"):
        # Different models need different input formatting and/or extra arguments
        prefix = args.prefix if args.prefix else args.padding_text # default = ""
        # print(f"prefix: {prefix}")
        # print(f"prompt: {prompt_text}")
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device) # encode prefix

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        print(f"\nshape of prefix: {len(encoded_prompt[0]), input_ids.shape}") 
        
        # contrastive decoding (papers)
        if args.do_sample == 'no' and (args.contrastive_decoding == 'student' or args.contrastive_decoding == 'earlystop' or args.contrastive_decoding == 'ngram') :
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                min_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                min_prob=args.min_prob,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_beams=args.num_beam,
                num_return_sequences=args.num_return_sequences,
                student_lm=student_lm,
                teacher_student=True,
                model_kwargs_student={}, 
                st_coef=args.st_coef,
                tokenizer=tokenizer, # analysis
                student_min_prob=args.student_min_prob,
                student_temperature=args.student_temperature,
                use_cap_student=(args.use_cap_student=='yes'), #cap student debug
                use_switch=(args.use_switch == 'yes')
            )
            print('student=smaller model')

        # max prob
        elif args.do_sample=='greedy' and args.contrastive_decoding == 'none':
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=False,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=False,
            model_kwargs_student={}, 
            st_coef=args.st_coef)
            print('greedy')

        # typical decoding
        elif args.do_sample == 'typical':
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            typical_p=0.95,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=False,
            model_kwargs_student={}, 
            st_coef=args.st_coef)
            print('typical sampling')
        
        # contrastive search 
        elif args.do_sample == 'contrastive_search_baseline':
            if 'gpt2' in args.model_name_or_path:
                prefix_text = prompt_text
                tokens = tokenizer.tokenize(prefix_text)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.LongTensor(input_ids).view(1,-1).to(model.model.device)
                beam_width, alpha, decoding_len = 4, 0.6, 256
            else:
                prefix_text = prompt_text
                tokens = tokenizer.tokenize(prefix_text)
                input_ids = tokenizer.convert_tokens_to_ids(tokens) # adds </s> to the beginning of every prompt
                input_ids = torch.LongTensor(input_ids).view(1,-1).to(model.model.device)
                beam_width, alpha, decoding_len = 5, 0.6, 256
            print(model.model.device, input_ids.device)
            # print(input_ids)
            output = model.fast_contrastive_search(input_ids=input_ids, beam_width=beam_width, 
                                                alpha=alpha, decoding_len=decoding_len,
                                                end_of_sequence_token_id = eos_token_id, early_stop = False) 
            # print("Output:\n" + 100 * '-')
            print(tokenizer.decode(output))
            print("" + 100 * '-')
            output_sequences = torch.tensor([output]).to(input_ids.device) 
            print('contrastive search baseline')
        
        # ========= my decoding ============
            
        else: # top-k (50) / nucleus (0.95) -> 실행할 때 args 설정
            output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            min_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=args.num_return_sequences,
            student_lm=student_lm,
            teacher_student=False,
            model_kwargs_student={}, 
            st_coef=args.st_coef)
            print(f'{args.do_sample}_sampling')

        print('analysis', output_sequences.shape, 'output.shape', input_ids.shape, 'input_ids.shape')
        
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for _, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {iidx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            # print(tokenizer.batch_decode(generated_sequence))
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_dict = format_out(total_sequence, prompt_text, generated_sequence, gold_ref=ref_lst[iidx][1])
            generated_sequences.append(generated_dict)
            print(total_sequence)

        generation_lst.append(generated_sequences)

    out_file(args, generation_lst)
        
if __name__ == "__main__":
    args = get_args() 
    main(args) 