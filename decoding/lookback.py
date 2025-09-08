import os
import numpy as np
import scipy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
import random
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

def plot_prob(list1, list2, beta, smooth_size, what):

    save_dir = "./img"
    save_path = os.path.join(save_dir, f"min_kld_{what}.png")
    x1 = np.arange(len(list1))
    x2 = np.arange(len(list2))
    y1 = np.array(list1)
    y2 = np.array(list2)
    y1_smooth = np.convolve(y1, np.ones(smooth_size)/smooth_size, mode="same")
    y2_smooth = np.convolve(y2, np.ones(smooth_size)/smooth_size, mode="same")
    y1_mean = np.mean(y1)
    y2_mean = np.mean(y2)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x1, y1_smooth, color="blue", label='before')
    plt.plot(x2, y2_smooth, color="red", label="after")
    plt.axhline(y=beta, color="gray", linestyle="--", label=f'beta = {beta}')
    plt.axhline(y=y1_mean, color="cornflowerblue", linestyle="--", label=f"Mean of Before: {y1_mean:.2f}")
    plt.axhline(y=y2_mean, color="orange", linestyle="--", label=f"Mean of After: {y2_mean:.2f}")
    
    plt.xlabel('Continuation Step')
    plt.ylabel('min kld')
    plt.title(f"Min KLD to {what}")    
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_overlapped(y, smooth_size, name):
    
    save_dir = "./img"
    save_path = os.path.join(save_dir, f"overlapped_{name}.png")
    x = np.arange(len(y))
    y = np.array(y)
    y_smooth = np.convolve(y, np.ones(smooth_size)/smooth_size, mode='same') # 각 값이 중간에 있을 때 smooth의 합/smooth size
    y_mean = np.mean(y)
    y_max = np.max(y)
    y_min = np.min(y)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y_smooth, color="blue")
    plt.axhline(y=y_mean, color="cornflowerblue", linestyle="--", label=f"Mean={y_mean:.1f}")
    plt.axhline(y=y_max, color='orange', linestyle='--', label=f"Max={y_max}")
    plt.axhline(y=y_min, color='forestgreen', linestyle='--', label=f"Min={y_min}")
    plt.axhline(y=5, color="gray", linestyle="--", label="Half=5")
    
    plt.xlabel('Continuation Step')
    plt.ylabel('overlapped count')
    plt.title('Count of Overlapped ids between Top-10 Prob and Top-10 PMI')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def without_prefix(model, input_ids):

    # attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids)
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
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            prob = F.softmax(logits/temperature, dim=-1)
            prefix_list.append(prob)
        next_token = prefix_ids[:, i].unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return prefix_list
    
def my_decoding(model, tokenizer, input_ids, max_new_tokens=256, window_size=5, alpha=0.5, beta=0.5, topk=10, temperature=1.0, device="cuda:0"):
    
    prob_list = [] # store the entire (logits, prob_ori)
    generated_tokens = input_ids.clone()
    prefix_list = prefix_prob(model, tokenizer, input_ids, temperature=temperature, device=device)
    window_list = [] # store only the window logits
    
    for i in range(max_new_tokens):

        print(f"token {i}")
        print("--"*10)

        with torch.no_grad():
            
            attention_mask = torch.ones_like(generated_tokens)
            outputs = model(input_ids=generated_tokens, attention_mask=attention_mask) 
            logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
            _, topk_logits, remove_indices = _top_k_sampling(logits, top_k=topk, temperature=temperature)
            # topk_logits[topk_logits == -float("inf")] = 0.0 # top-k logits 외에는 모두 0으로 만들기

            prob = F.softmax(logits/temperature, dim=-1) # original probability
            prev_list = prefix_list + prob_list # the entire probablity distribution including prefix
            
            if window_list:
                # detect repetition in previous list
                min_kld = float("inf")
                for p, _prev_prob in enumerate(prev_list):
                    KLD = kld(prob, _prev_prob)
                    if min_kld > KLD:
                        min_kld = KLD
                noninf_ids = torch.nonzero(topk_logits != -float("inf"), as_tuple=True)[1] # top-k logits의 id만 모아놓은 1차원 텐서
                noninf_vals = topk_logits[0, noninf_ids].unsqueeze(0) # top-k logits만 모아놓은 2차원 텐서
                print(f"noninf_ids: {noninf_ids}")
                kld_list = []
                for k in range(len(noninf_ids)):
                    token_id = noninf_ids[k].unsqueeze(0).unsqueeze(0)
                    temp_ids = torch.cat([generated_tokens, token_id], dim=-1)
                    v_logits = without_prefix(model=model, input_ids=temp_ids)
                    v_prob = F.softmax(v_logits, dim=-1)
                    
                    min_kld_prefix = float("inf")
                    for _prefix_prob in prefix_list:
                        KLD = kld(v_prob, _prefix_prob)
                        if min_kld_prefix > KLD:
                            min_kld_prefix = KLD
                    kld_list.append(min_kld_prefix)
                
                print(f"length of kld_list: {len(kld_list)}")
                kld_tensor = torch.tensor(kld_list).unsqueeze(0) # 2차원 텐서
                print(f"kld_tensor: {kld_tensor}")
                print(f"kld_tensor_len: {kld_tensor.shape}")

                # _, window_logits = remove_ids(window_list[0], remove_indices, temperature=temperature)
                # window_logits[window_logits == -float("inf")] = 0.0
                '''
                top-k로 필터링하고 argmax id를 구하면 0~9의 id가 선택되기 때문에 실제 id 중에서 0~9번 id가 선택되어 잘못 next_token이 생성됨 -> top-k로 필터링해도 원래 크기 (1, vocab_size) 유지해야 함
                '''
                if min_kld <= beta:
                    final_logits = -kld_tensor
                    print(f"repetition penalized at {i}\n")
                    final_prob = F.softmax(final_logits, dim=-1)
                    next_token_idx = torch.argmax(final_prob, dim=-1)
                    print(f"next_token_idx: {next_token_idx}")
                    next_token = noninf_ids[next_token_idx.item()].to(device)   
                    next_token = next_token.unsqueeze(0)
                else:
                    final_logits = logits
                    final_prob = F.softmax(final_logits, dim=-1)
                    next_token = torch.argmax(final_prob, dim=-1)
            else:
                next_token = torch.argmax(prob, dim=-1)
            
            print(f"next_token_id: {next_token}")
            print(f"final decode: {tokenizer.convert_ids_to_tokens(next_token.item())}\n")

            if next_token.item() == tokenizer.eos_token_id:
                break
            
            _next_token = next_token.unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, _next_token], dim=-1)
            prob_list.append(prob)
            window_list = extend_window(window_list, logits, window_size)

    return generated_tokens

def main():
    
    model_name = "openai-community/gpt2-xl"
    
    prompt='''For the first time ever, a person is born with a genuine superpower. They proceed to live out their entire life without noticing or realizing it.'''
    # prompt = prompt.split()[:32]
    # prompt = " ".join(prompt)
    device = "cuda:3"
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
                                beta=1.6, 
                                topk=5,
                                temperature=1.0,
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