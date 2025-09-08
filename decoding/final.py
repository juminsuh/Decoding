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

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42)

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None) # float16으로 하면 Underflow돼서 nan값이 생겨서 에러남
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=config.device_map, cache_dir=None)
    return tokenizer, model

def greedy(model, tokenizer, input_ids, max_new_tokens=256):

    output_ids = model.generate(input_ids,
                                max_new_tokens=max_new_tokens,
                                do_sample=False)

    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def top_k(model, tokenizer, input_ids, max_new_tokens=256, temperature=0.7, top_k=10):
    
    output_ids = model.generate(input_ids,
                                max_new_tokens=max_new_tokens,
								temperature=temperature,
								top_k=top_k,
								do_sample=True)
    
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def top_p(model, tokenizer, input_ids, max_new_tokens=256, temperature=0.7, top_p=0.95):

    output_ids = model.generate(input_ids,
                                max_new_tokens=max_new_tokens,
								temperature=temperature,
								top_p=top_p,
								do_sample=True)
    
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def beam_search_decoding(model, tokenizer, input_ids, max_new_tokens=256,  beam_size=5):

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        early_stopping=True,         
        do_sample=False,              
    )
    
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def _top_k_sampling(logits, top_k=top_k, temperature=0.7):
    _, topk_indices = torch.topk(logits, top_k, dim=-1)
    top_ids = topk_indices[0]
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask[0, top_ids] = True  # (1, vocab_size): top-k indices

    remove_indices = ~keep_mask

    filtered_logits = logits.clone()
    filtered_logits[remove_indices] = -float("inf")
    probs = F.softmax(filtered_logits/temperature, dim=-1)
    return probs, remove_indices

def remove_ids(logits, indices, filter_value=-float("inf"), temperature=0.7):
    filtered_logits = logits.clone()
    filtered_logits[indices] = filter_value
    prob = F.softmax(filtered_logits/temperature, dim=-1)
    return prob

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
    plt.axhline(y=50, color="gray", linestyle="--", label="Half=50")
    
    plt.xlabel('Continuation Step')
    plt.ylabel('overlapped percentage (%)')
    plt.title('Percentage of Overlapped ids between Top-10 Prob and Top-10 PMI')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def without_prefix(model, generated_ids):
    with torch.no_grad():
        outputs = model(generated_ids)
        logits = outputs.logits[:, -1, :]
    
    return logits

def prefix_prob(model, tokenizer, prefix_ids, temperature, device):
    prefix_list = []
    logits_list = []
    prefix_ids = prefix_ids.clone()
    if tokenizer.bos_token is not None:
        input_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device) # GPT-2 doesn't need bos
    
    for i in range(prefix_ids.shape[1]):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            prob = F.softmax(logits/temperature, dim=-1)
            prefix_list.append(prob)
            logits_list.append(logits)
        next_token = prefix_ids[:, i].unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return prefix_list, logits_list
    
def my_decoding(model, tokenizer, input_ids, max_new_tokens=256, window_size=5, alpha=0.5, beta=0.5, topk=10, temperature=1.0, device="cuda:0"):
    
    prob_list = [] # store the entire (logits, prob_ori)
    generated_tokens = input_ids.clone()
    min_kld_list = []
    my_kld_list = []
    # overlap_list_p = []
    overlap_list_l = []
    prefix_list, logits_list = prefix_prob(model, tokenizer, input_ids, temperature=temperature, device=device)
    window_list = logits_list[-5:] # store only the window logits
    min_kld_pre_list = []
    my_kld_pre_list = []
        
    for i in range(max_new_tokens):

        print(f"token {i}")
        print("--"*10)

        with torch.no_grad():

            outputs = model(generated_tokens) 
            logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
            prob_ori = F.softmax(logits/temperature, dim=-1) # original probability
            prob, remove_indices = _top_k_sampling(logits, top_k=topk, temperature=temperature)
            prev_list = prefix_list + prob_list # the entire probablity distribution including prefix

            # ppmi 
            # if i == 0:
            #     if tokenizer.bos_token is not None:
            #         gen_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)
            #     else:
            #         gen_ids = torch.tensor([[tokenizer.bos_token_id]], device=device) # GPT-2 doesn't need bos
            # else:
            #     prefix_len = input_ids.shape[1] # length of prefix
            #     gen_ids = generated_tokens.clone()
            #     gen_ids = gen_ids[:, prefix_len:].unsqueeze(0)

            # gen_logits = without_prefix(model, gen_ids) # logits computed without prefix
            # gen_prob = remove_ids(gen_logits, remove_indices)
            # ppmi = torch.log((prob+1e-8)/(gen_prob+1e-8)) # pmi between prefix and current token v
            # ppmi = torch.max(ppmi, torch.tensor(0.0, device=ppmi.device))

            # detect repetition in previous list
            min_kld = float("Inf")
            min_kld_idx = 0
            for p, _prev_prob in enumerate(prev_list):
                KLD = kld(prob_ori, _prev_prob)
                if min_kld > KLD:
                    min_kld = KLD
                    min_kld_idx = p
            print(f"min_kld_idx: {min_kld_idx}")
            print(f"min_kld: {min_kld}")
            min_kld_list.append(min_kld)
            
            min_kld_pre = float("Inf")
            min_kld_idx_pre = 0
            for p, _prefix_prob in enumerate(prefix_list):
                KLD = kld(prob_ori, _prefix_prob)
                if min_kld_pre > KLD:
                    min_kld_pre = KLD
                    min_kld_idx_pre = p
            print(f"min_kld_idx_pre: {min_kld_idx_pre}")
            print(f"min_kld_pre: {min_kld_pre}\n") 
            min_kld_pre_list.append(min_kld_pre)

            # lpmi
            window_prob = remove_ids(window_list[0], remove_indices, temperature=temperature)     
            lpmi = torch.log((prob+1e-8)/(window_prob+1e-8)) # pmi between window and current token v
            lpmi = torch.max(lpmi, torch.tensor(0.0, device=lpmi.device))
            lpmi_vals, lpmi_indices = torch.topk(lpmi, topk, dim=-1)
            print(f"lpmi_vals: {lpmi_vals[0]}") # 걸러진 애들 중에서 top-k lpmi
            print(f"lpmi_indices: {lpmi_indices[0]}\n")

            if min_kld <= beta:
                final_logits = logits - alpha * lpmi 
                print(f"repetition penalized at {i}\n")
            else:
                final_logits = logits + alpha * lpmi
            
            final_prob = F.softmax(final_logits, dim=-1)

            _window_prob = F.softmax(window_list[0]/temperature, dim=-1)
            _lpmi = torch.log((prob_ori+1e-8)/(_window_prob+1e-8))
            _lpmi = torch.max(_lpmi, torch.tensor(0.0, device=_lpmi.device))
            topk_lpmi, topk_indices = torch.topk(_lpmi, topk, dim=-1)
            lpmi_ava = topk_indices[0] # 걸러지지 않았을 때 top-k lpmi
            print(f"topk_lpmi: {topk_lpmi[0]}")
            print(f"lpmi_ava: {lpmi_ava}")
            
            prob_ava = torch.nonzero(prob != 0, as_tuple=True)[1] # top-k prob indices
            sorted_ids = prob_ava[prob[0, prob_ava].argsort(descending=True)]
            print(f"prob_ava: {sorted_ids}\n")
            
            # overlapped_p = torch.isin(ppmi_ava, prob_ava)
            # overlapped_p = overlapped_p.sum().item()
            # overlap_list_p.append(overlapped_p*10)

            overlapped_l = torch.isin(lpmi_ava, prob_ava)
            overlapped_l = overlapped_l.sum().item()
            overlap_list_l.append(overlapped_l)

            my_kld = float("Inf")
            my_kld_idx = 0
            for p, _prev_prob in enumerate(prev_list):
                KLD = kld(final_prob, _prev_prob)
                if my_kld > KLD:
                    my_kld = KLD
                    my_kld_idx = p
            my_kld_list.append(my_kld)
            print(f"my_kld_idx: {my_kld_idx}")
            print(f"my_kld: {my_kld}")
            
            my_kld_pre = float("Inf")
            my_kld_idx_pre = 0
            for p, _prefix_prob in enumerate(prefix_list):
                KLD = kld(final_prob, _prefix_prob)
                if my_kld_pre > KLD:
                    my_kld_pre = KLD
                    my_kld_idx_pre = p
            print(f"my_kld_idx_pre: {my_kld_idx_pre}")
            print(f"my_kld_pre: {my_kld_pre}\n")
            my_kld_pre_list.append(my_kld_pre)
        
            next_token = torch.argmax(final_prob, dim=-1)
            print(f"final decode: {tokenizer.convert_ids_to_tokens(next_token.item())}\n")

            if next_token.item() == tokenizer.eos_token_id:
                break
            
            _next_token = next_token.unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, _next_token], dim=-1)
            prob_list.append(prob_ori)
            window_list = extend_window(window_list, logits, window_size)

    # plot_overlapped(overlap_list_p, smooth_size=10, name="prefix")
    plot_overlapped(overlap_list_l, smooth_size=10, name="window")
    plot_prob(min_kld_list, my_kld_list, beta, smooth_size=10, what="previous")
    plot_prob(min_kld_pre_list, my_kld_pre_list, beta, smooth_size=10, what="prefix")
    return generated_tokens

def main():
    
    model_name = "facebook/opt-6.7b"
    
    prompt='''The game takes place during the Second Europan War . Gallian Army Squad 422 , also known as " The Nameless " , are a penal military unit composed of criminals , foreign deserters , and military offenders whose real names are erased from the records and thereon officially referred to by numbers . Ordered by the Gallian military to perform the most dangerous missions that the Regular Army and Militia will not do , they are nevertheless up to the task , exemplified by their motto , Altaha Abilia , meaning " Always Ready . " The three main characters are No.7 Kurt Irving , an army officer falsely accused of treason who wishes to redeem himself ; Ace No.1 Imca , a female Darcsen heavy weapons specialist who seeks revenge against the Valkyria who destroyed her home ; and No.13 Riela Marcellis , a seemingly jinxed young woman who is unknowingly a descendant of the Valkyria . Together with their fellow squad members , these three are tasked to fight against a mysterious Imperial unit known as Calamity Raven , consisting of mostly Darcsen soldiers .'''
    prompt = prompt.split()[:32]
    prompt = " ".join(prompt)
    device = "cuda:3"
    tokenizer, model = load_model(model_name)
    # model.to(device)
    prompt_input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = prompt_input["input_ids"].to(device)
    model.eval()

    output_tokens = my_decoding(model,
                                tokenizer,
                                input_ids,
                                max_new_tokens=256,
                                window_size=5, 
                                alpha=3.0,
                                beta=1.6, 
                                topk=10,
                                temperature=1.0,
                                device=device) 
    my_decoded_tokens = tokenizer.decode(output_tokens[0][input_ids.shape[1]:], skip_special_tokens=True)
    greedy_output = greedy(model, tokenizer, input_ids)
    topk_output = top_k(model, tokenizer, input_ids)
    topp_output = top_p(model, tokenizer, input_ids)
    beam_output = beam_search_decoding(model, tokenizer, input_ids)

    print(f"prompt: {prompt}\n\n")
    print("A:\n", greedy_output)
    print("__" * 50)
    print("B:\n", topk_output)
    print("__" * 50)
    print("C:\n", topp_output)
    print("__" * 50)
    print("D:\n", beam_output)
    print("__" * 50)
    print("E:\n", my_decoded_tokens)
    
if __name__ == "__main__":
    main()