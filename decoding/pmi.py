import os
import numpy as np
import scipy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import config
# from evaluate import load

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

def _top_k_sampling(logits: torch.Tensor, top_k: int = 10):
    _, topk_indices = torch.topk(logits, top_k, dim=-1)
    top_ids = topk_indices[0]
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask[0, top_ids] = True  # (1, vocab_size): top-k indices

    remove_indices = ~keep_mask

    filtered_logits = logits.clone()
    filtered_logits[remove_indices] = -float("inf")
    probs = F.softmax(filtered_logits, dim=-1)
    return probs, remove_indices

def remove_ids(logits, indices, filter_value=-float("inf")):
    filtered_logits = logits.clone()
    filtered_logits[indices] = filter_value
    prob = F.softmax(filtered_logits, dim=-1)
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
    
def plot_overlapped(y, smooth_size):
    
    save_dir = "./img"
    save_path = os.path.join(save_dir, "overlapped.png")
    x = np.arange(len(y))
    y = np.array(y)
    y_smooth = np.convolve(y, np.ones(smooth_size)/smooth_size, mode='same') # 각 값이 중간에 있을 때 smooth의 합/smooth size
    y_mean = np.mean(y)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y_smooth, color="blue")
    plt.axhline(y=y_mean, color="gray", linestyle="--", label=f"Mean={y_mean:.1f}")
    plt.axhline(y=50, color="orange", linestyle="--", label="Half=50")
    
    plt.xlabel('Continuation Step')
    plt.ylabel('overlapped percentage (%)')
    plt.title('Percentage of Overlapped ids between Top-10 Prob and Top-10 PMI')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def prefix_prob(model, tokenizer, prefix_ids, device):
    prefix_list = []
    prefix_ids = prefix_ids.clone()
    if tokenizer.bos_token is not None:
        input_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device) # GPT-2 doesn't need bos
    
    for i in range(prefix_ids.shape[1]):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            prefix_list.append(prob)
        next_token = prefix_ids[:, i].unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return prefix_list

def my_decoding(model, tokenizer, input_ids, max_new_tokens=256, window_size=5, alpha=0.5, beta=0.5, temperature=0.7, device="cuda:0"):
    
    prob_list = [] # store the entire (logits, prob_ori)
    generated_tokens = input_ids.clone()
    window_list = [] # store only the window logits
    min_kld_list = []
    my_kld_list = []
    overlap_list = []
    prefix_list = prefix_prob(model, tokenizer, input_ids, device)
    min_kld_pre_list = []
    my_kld_pre_list = []
        
    for i in range(max_new_tokens):

        print(f"token {i}")
        print("--"*10)
        # attention_mask = torch.ones_like(generated_tokens)

        with torch.no_grad():

            outputs = model(generated_tokens) 
            logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)
            prob_ori = F.softmax(logits/temperature, dim=-1) # original probability
            prob, remove_indices = _top_k_sampling(logits)
            # next_token_ori = torch.argmax(prob_ori, dim=-1)
            # print(f"original prob: {prob_ori[0, next_token_ori].item()}")
            # print(f"next_token_ori: {next_token_ori}")
            # print(f"final decode_ori: {tokenizer.convert_ids_to_tokens(next_token_ori.item())}\n")
            
            if window_list:
                # cppmi
                window_prob = remove_ids(window_list[0], remove_indices)     
                cpmi = torch.log((prob+1e-8)/(window_prob+1e-8))
                # print(f"cpmi: {cpmi}")
                # cpmi = cpmi*(len(window_list)/window_size)
                cppmi = torch.max(cpmi, torch.tensor(0.0, device=cpmi.device))
                
                # confirm 
                _cpmi = torch.log((prob_ori+1e-8)/(window_list[0]+1e-8))
                _cppmi = torch.max(_cpmi, torch.tensor(0.0, device=_cpmi.device))
                _, topk_indices = torch.topk(_cppmi, 10, dim=-1)
                pmi_ava = topk_indices[0] # top-k cpmi indices
                print(f"pmi_ava: {pmi_ava}")
                
                prob_ava = torch.nonzero(prob != 0, as_tuple=True)[1] # top-k prob indices
                sorted_ids = prob_ava[prob[0, prob_ava].argsort(descending=True)]
                # bos_ava = torch.nonzero(bos_prob, as_tuple=True)[1]    
                print(f"prob_ava: {sorted_ids}")
                # print(f"bos_ava:{bos_ava}")
                
                overlapped = torch.isin(pmi_ava, prob_ava)
                overlapped = overlapped.sum().item()
                overlap_list.append(overlapped*10)
                
                # pmi_list.append(cppmi)
                # print(f"cppmi: {cppmi}")
                # prob_ava = torch.nonzero(prob != 0, as_tuple=True)[1]
                # cpmi_ava = torch.nonzero(cppmi != 0, as_tuple=True)[1]
                # print(f"prob_ava: {prob_ava}")
                # print(f"cpmi_ava: {cpmi_ava}")
                
                # rppmi
                # detect repetition with history
                min_kld = float("Inf")
                min_kld_idx = 0
                # min_logits = 0
                for p, _prev_prob in enumerate(prob_list):
                    KLD = kld(prob_ori, _prev_prob)
                    if min_kld > KLD:
                        min_kld = KLD
                        min_kld_idx = p
                        # min_logits = _prev_logits # logits which have the least kld with the current prob. dist.
                # min_kld_list.append(min_kld)
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
                print(f"min_kld_pre: {min_kld_pre}") 
                min_kld_pre_list.append(min_kld_pre)
                
                if min_kld <= beta: # if repetition: rppmi
                    # min_kld_list.append(min_kld)
                    # rppmi = pmi_list[p-1]
                    # prefix_prob = remove_ids(prefix_logits, remove_indices)
                    # min_prob = remove_ids(min_logits, remove_indices)

                    # rpmi = torch.log((prob+1e-8)/(prob + 1e-8)) 
                    # rppmi = torch.max(rpmi, torch.tensor(0.0, device=rpmi.device))
                    # print(f"rppmi: {rppmi}")
                    final_logits = logits - alpha * cppmi
                    print(f"repetition penalized at {i}")                    
                else:
                    final_logits = logits + alpha * cppmi
                    
                final_prob = F.softmax(final_logits, dim=-1)
                my_kld = float("Inf")
                my_kld_idx = 0
                for p, _prev_prob in enumerate(prob_list):
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
                print(f"my_kld_pre: {my_kld_pre}")
                my_kld_pre_list.append(my_kld_pre)
            
                
                # top_k_prob, _ = _top_k_sampling(final_prob)
                # next_token = torch.multinomial(top_k_prob, num_samples=1)
                next_token = torch.argmax(final_prob, dim=-1)
                
                # print(f"final prob: {final_prob[0, next_token].item()}")
                # print(f"next_token: {next_token.item()}")
                print(f"final decode: {tokenizer.convert_ids_to_tokens(next_token.item())}\n")
                
            else: # if there's no window yet (if i == 0)
                next_token = torch.argmax(prob, dim=-1)
                # top_k_prob, _ = _top_k_sampling(prob_ori)
                # next_token = torch.multinomial(top_k_prob, num_samples=1)
                # prefix_logits = logits
                # prefix_prob = prob_ori
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            _next_token = next_token.unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, _next_token], dim=-1)
            prob_list.append(prob_ori)
            window_list = extend_window(window_list, prob_ori, window_size)
    
    plot_overlapped(overlap_list, smooth_size=10)
    plot_prob(min_kld_list, my_kld_list, beta, smooth_size=10, what="history")
    plot_prob(min_kld_pre_list, my_kld_pre_list, beta, smooth_size=10, what="prefix")
    return generated_tokens

def main():
    
    model_name = "facebook/opt-6.7b"
    
    prompt='''For the first time ever, a person is born with a genuine superpower. They proceed to live out their entire life without noticing or realizing it.'''
    prompt = prompt.split()[:32]
    prompt = " ".join(prompt)
    # golden = "Valkyria Chronicles III (Japanese: Senjō no Valkyria 3), commonly referred to as Valkyria Chronicles III outside Japan, is a tactical role‑playing video game … the story runs parallel to the first game and follows the ‘Nameless’, a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit ‘Calamity Raven’."
    # prompt = f'''Summarize the following article.\n\n{article}\n\nSummarization: '''
    # prompt = f'''Write the continued text of the following article.\n\n{article}\n\nContinued text: '''
    device = "cuda:3"
    tokenizer, model = load_model(model_name)
    # model.to(device)
    # dispatch_model(model, device_map=config.device_map)
    # print(model.hf_device_map)

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
                                temperature=0.7,
                                device=device) 
    my_decoded_tokens = tokenizer.decode(output_tokens[0][input_ids.shape[1]:], skip_special_tokens=True)
    greedy_output = greedy(model, tokenizer, input_ids)
    topk_output = top_k(model, tokenizer, input_ids)
    topp_output = top_p(model, tokenizer, input_ids)
    beam_output = beam_search_decoding(model, tokenizer, input_ids)
    
    # mauve = load('mauve') # 단일 예제에 대한 평가 지표가 아니라, 서로 다른 여러 개의 predictions - golden 사이의 일치도를 전반적으로 측정하는 평가 지표
    # predictions = [greedy_output, topk_output, topp_output, beam_output, my_decoded_tokens]
    # references = [golden]*5
    # mavue_results = mauve.compute(predictions=predictions, references=references)
    
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
    
    # print(f"mauve_score: {mavue_results}")
    
if __name__ == "__main__":
    main()