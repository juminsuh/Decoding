import os
import numpy as np
import scipy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
import random
import math
import matplotlib.pyplot as plt
import config
from scipy import stats
from accelerate import dispatch_model

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42)

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None) # float16으로 하면 Underflow돼서 nan값이 생겨서 에러남
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=None)
    # print(len(model.model.decoder.layers))
    if model_name == 'facebook/opt-6.7b':
        dispatch_model(model, device_map=config.device_map)
    if model_name == 'openai-community/gpt2-xl':
        model.to(device)
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

def greedy(model, tokenizer, input, max_new_tokens=256):

    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    output_ids = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=False)

    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def top_k(model, tokenizer, input, max_new_tokens=256, temperature=0.7, top_k=10):

    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    output_ids = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
								temperature=temperature,
								top_k=top_k,
								do_sample=True)
    
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def top_p(model, tokenizer, input, max_new_tokens=256, temperature=0.7, top_p=0.95):

    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    output_ids = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
								temperature=temperature,
								top_p=top_p,
								do_sample=True)
    
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def beam_search_decoding(model, tokenizer, input, max_new_tokens=256,  beam_size=5):

    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        early_stopping=True,         
        do_sample=False,              
    )
    
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def _top_k_sampling(logits, top_k=10, temperature=0.7):
    _, topk_indices = torch.topk(logits, top_k, dim=-1)
    top_ids = topk_indices[0]
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask[0, top_ids] = True  # (1, vocab_size): top-k indices

    remove_indices = ~keep_mask

    filtered_logits = logits.clone()
    filtered_logits[remove_indices] = -float("inf")
    probs = F.softmax(filtered_logits/temperature, dim=-1)
    return probs, filtered_logits, remove_indices

def remove_ids(logits, indices, filter_value=-float("inf"), temperature=0.7):
    filtered_logits = logits.clone()
    filtered_logits[indices] = filter_value
    prob = F.softmax(filtered_logits/temperature, dim=-1)
    return prob, filtered_logits

def extend_window(window_list, logits, window_size):
    window_list.append(logits)
    if len(window_list) > window_size:
        window_list.pop(0)
    return window_list

def kld(p, q, eps=1e-8): # p, q: post-softmaxed probability distribution
    p = p.to("cpu").numpy().astype(np.float32)
    q = q.to("cpu").numpy().astype(np.float32)    
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    
    kl = (p * (np.log(p) - np.log(q))).sum()
    return kl

def without_prefix(model, input_ids):
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
    
    return logits

def prefix_prob(model, tokenizer, prefix_ids, temperature, device):
    prefix_list = []
    prefix_ids = prefix_ids.clone().to(device)
    if tokenizer.bos_token is not None:
        input_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)
        iter = prefix_ids.shape[1]
    else:
        input_ids = prefix_ids[:, :1] # GPT-2 doesn't have BOS
        iter = prefix_ids.shape[1] - 1
    
    for i in range(iter):
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            prob = F.softmax(logits/temperature, dim=-1)
            prefix_list.append(prob)
        next_token = prefix_ids[:, i].unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return prefix_list

def plot_entropy(entropies, type):
    """
    entropies: list of float — conditional entropy per decoding step
    save_path: str or None — if given, the plot will be saved to this path
    """
    save_dir = "./img"
    save_path = os.path.join(save_dir, f"entropy-{type}.png")
    steps = np.arange(1, len(entropies) + 1)
    mean_entropy = np.mean(entropies)

    plt.figure(figsize=(10, 4))
    plt.plot(steps, entropies, marker='o', label='Entropy per step')
    plt.axhline(y=mean_entropy, color='r', linestyle='--', label=f'Mean Entropy = {mean_entropy:.4f}')
    plt.xlabel('Generation Step')
    plt.ylabel('Conditional Entropy')
    plt.title('Entropy over Decoding Steps')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def my_decoding(model, tokenizer, input_ids, max_new_tokens=256, window_size=5, alpha=0.5, beta=0.5, topk=10, temperature=1.0, device="cuda:0"):
    
    prob_list = [] # store the entire (logits, prob_ori)
    generated_tokens = input_ids.clone()
    prefix_list = prefix_prob(model, tokenizer, input_ids, temperature=temperature, device=device)
    window_list = [] # store only the window logits
    entropies = []
    
    for i in range(max_new_tokens):

        print(f"token {i}")
        print("--"*10)

        with torch.no_grad():

            attention_mask = torch.ones_like(generated_tokens)
            outputs = model(input_ids=generated_tokens, attention_mask=attention_mask) 
            logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
            # _, topk_logits, remove_indices = _top_k_sampling(logits, top_k=topk, temperature=temperature) # top-k만 정상적인 로짓, 나머지는 -inf

            prob = F.softmax(logits/temperature, dim=-1) # original probability
            # prev_list = prefix_list + prob_list # the entire probablity distribution including prefix
            
            if prob_list:
                # detect repetition in previous list
                # min_kld = float("inf")
                # min_kld_idx = 0
                # min_kld_logits = 0
                # for p, (_prev_prob, _prev_logits) in enumerate(prob_list):
                #     KLD = kld(prob, _prev_prob)
                #     if min_kld > KLD:
                #         min_kld = KLD
                #         min_kld_idx = p
                #         min_kld_logits = _prev_logits
                # print(f"min_kld: {min_kld}")
                # print(f"min_kld_idx: {min_kld_idx}")
                
                # compute conditional entropy
                # entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1).unsqueeze(0) # 2차원 텐서
                # entropies.append(entropy.squeeze(0).item())

                # adaptive plausibility constraint
                # gamma = 0.1
                # max_prob, _ = torch.max(prob, dim=-1)
                # v_head = torch.nonzero(prob >= gamma * max_prob.item(), as_tuple=True)[1] # threshold보다 큰 확률값을 갖는 인덱스; 1차원 텐서
                # topk_logits = logits[:, v_head] # threshold보다 큰 확률값을 갖는 로짓들; 2차원 텐서
                # window_logits = window_list[0][:, v_head] # 대응되는 로짓들; 2차원 텐서
                # print(f"v_head: {v_head}")
                # print(f"topk_logits: {topk_logits}")
                # print(f"window_logits: {window_logits}")
                
                
                topk_logits, topk_logits_indices = torch.topk(logits, topk, dim=-1) # 2차원 텐서
                print(f"topk_logits: {topk_logits}")
                print(f"topk_logits_indices: {topk_logits_indices}")

                topk_indices = topk_logits_indices.squeeze(0)
                window_logits = window_list[0][1][:, topk_indices]
                print(f"window_logits: {window_logits}\n")

                topk_prob = prob[:, topk_indices] # 2차원 텐서
                window_prob = window_list[0][0][:, topk_indices] # 2차원 텐서

                # 현재 타임 스텝의 pmi
                current_pmi = torch.log((topk_prob+1e-8)/(window_prob+1e-8)) # 2차원 텐서
                pmi_sum = current_pmi.sum(dim=-1).item() # 실수
                print(f"pmi_sum: {pmi_sum}")

                maxum_idx = 0
                maxum_val = -float("inf")
                maxum_logits = 0
                maxum_logits_ = 0
                # minxum_idx = 0
                # minxum_val = float("inf")
                for t, (_prev_prob, _prev_logits) in enumerate(prob_list):
                    _topk_prob = _prev_prob[:, topk_indices]
                    if t <= 4:
                        _window_prob = prob_list[0][0][:, topk_indices]
                        _maxum_logits = prob_list[0][1][:, topk_indices]
                    else:
                        _window_prob = prob_list[t-window_size][0][:, topk_indices]
                        _maxum_logits = prob_list[t-window_size][1][:, topk_indices]

                    compare_pmi = torch.log((_topk_prob+1e-8)/(_window_prob+1e-8)) # previous pmi
                    compare_pmi_sum = compare_pmi.sum(dim=-1).item() # float; previous pmi의 합
                    # print(f"compare_pmi_sum: {compare_pmi_sum}")

                    if compare_pmi_sum > maxum_val:
                        maxum_idx = t
                        maxum_val = compare_pmi_sum
                        maxum_logits = _prev_logits[:, topk_indices]
                        maxum_logits_ = _maxum_logits
                    # if compare_pmi_sum < minxum_val:
                    #     minxum_idx = t
                    #     minxum_val = compare_pmi_sum

                print(f"maxum_idx: {maxum_idx}")
                print(f"maxum_val: {maxum_val}")
                print(f"maxum_logits: {maxum_logits}")
                print(f"maxum_logits_: {maxum_logits_}\n")
                # print(f"minxum_idx: {minxum_idx}")
                # print(f"minxum_val: {minxum_val}\n")

                # topk_indices = topk_logits_indices.squeeze(0)
                # window_logits = min_kld_logits[:, topk_indices]
                # print(f"window_logits: {window_logits}")
                
                # noninf_ids = torch.nonzero(topk_logits != -float("inf"), as_tuple=True)[1] # top-k logits의 id만 모아놓은 1차원 텐서
                # noninf_vals = topk_logits[0, noninf_ids].unsqueeze(0) # top-k logits만 모아놓은 2차원 텐서
                # print(f"noninf_ids: {noninf_ids}")
                # print(f"noninf_vals: {noninf_vals}\n")
                
                # _, window_logits = remove_ids(window_list[0], remove_indices, temperature=temperature)
                # noninf_window_ids = torch.nonzero(window_logits != -float("inf"), as_tuple=True)[1] # top-k logits의 id만 모아놓은 1차원 텐서
                # noninf_window_vals = window_logits[0, noninf_window_ids].unsqueeze(0) # top-k logits만 모아놓은 2차원 텐서 
                # print(f"noninf_window_ids: {noninf_window_ids}")
                # print(f"noninf_window_vals: {noninf_window_vals}\n")

                # generated_only = generated_tokens[:, input_ids.shape[1]:]# 생성된 텍스트의 id만 담은 2차원 텐서
                # print(f"length of generated_only: {generated_only.shape}")
                # prefix_logits = without_prefix(model, generated_only)
                # _, prefix_logits = remove_ids(prefix_logits, remove_indices, temperature=temperature)
                # noninf_prefix_ids = torch.nonzero(prefix_logits != -float("inf"), as_tuple=True)[1] # top-k logits의 id만 모아놓은 1차원 텐서
                # noninf_prefix_vals = prefix_logits[0, noninf_prefix_ids].unsqueeze(0) # top-k logits만 모아놓은 2차원 텐서 
                # print(f"noninf_prefix_ids: {noninf_prefix_ids}")
                # print(f"noninf_prefix_vals: {noninf_prefix_vals}\n")

                # window_logits = window_list[0]
            
                '''
                top-k로 필터링하고 argmax id를 구하면 0~k-1의 id가 선택되기 때문에 실제 id 중에서 0~k-1번 id가 선택되어 잘못 next_token이 생성됨
                '''
                if maxum_val >= pmi_sum:     
                    # final_logits = topk_logits - alpha * maxum_logits + alpha * maxum_logits_  
                    final_logits = (1-alpha) * topk_logits + alpha * window_logits
                    print(f"repetition\n")
                else:
                    final_logits = (1+alpha) * topk_logits - alpha * window_logits
                    print(f"upweight\n")

                print(f"final_logits: {final_logits}")
                final_prob = F.softmax(final_logits, dim=-1) # top-k 중에서만 softmax
                next_token_idx = torch.argmax(final_prob, dim=-1) # top-k 중에서 최대확률을 갖는 인덱스; 1차원 텐서
                print(f"next_token_idx: {next_token_idx}")
                next_token = topk_indices[next_token_idx.item()]
                next_token = next_token.unsqueeze(0)
            else:
                next_token = torch.argmax(prob, dim=-1)
            
            print(f"next_token_id: {next_token}")
            print(f"final decode: {tokenizer.convert_ids_to_tokens(next_token.item())}\n")

            if next_token.item() == tokenizer.eos_token_id:
                break
            
            _next_token = next_token.unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, _next_token], dim=-1)
            prob_list.append((prob, logits))
            window_list = extend_window(window_list, (prob, logits), window_size)

    # plot_entropy(entropies=entropies, type="my")
    return generated_tokens

def main():
    
    model_name = "facebook/opt-13b"
    
    prompt='''In a preview of the TGS demo , Ryan Geddes of IGN was left excited as to where the game would go after completing the demo , along with enjoying the improved visuals over Valkyria Chronicles II . Kotaku 's Richard Eisenbeis was highly positive about the game , citing is story as a return to form after Valkyria Chronicles II and its gameplay being the best in the series . His main criticisms were its length and gameplay repetition , along with expressing regret that it would not be localized .'''
    prompt = prompt.split()[:32]
    prompt = " ".join(prompt)
    device = "cuda:0"
    tokenizer, model = load_model(model_name, device)
    prompt_input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = prompt_input["input_ids"].to(device)
    model.eval()

    output_tokens = my_decoding(model,
                                tokenizer,
                                input_ids,
                                max_new_tokens=256,
                                window_size=5, 
                                alpha=0.5,
                                beta=1.0, 
                                topk=5,
                                temperature=0.7,
                                device=device) 
    my_decoded_tokens = tokenizer.decode(output_tokens[0][input_ids.shape[1]:], skip_special_tokens=True)
    print("E:\n", my_decoded_tokens, "\n")
    greedy_output = greedy(model, tokenizer, prompt_input)
    topk_output = top_k(model, tokenizer, prompt_input)
    topp_output = top_p(model, tokenizer, prompt_input)
    beam_output = beam_search_decoding(model, tokenizer, prompt_input)
    print(f"prompt: {prompt}\n\n")
    print("A:\n", greedy_output)
    print("__" * 50)
    print("B:\n", topk_output)
    print("__" * 50)
    print("C:\n", topp_output)
    print("__" * 50)
    print("D:\n", beam_output)
    print("__" * 50)
    
    
if __name__ == "__main__":
    main()