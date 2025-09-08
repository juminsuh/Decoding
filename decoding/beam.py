import os
import numpy as np
import scipy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

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
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=None, use_fast=False)
    return tokenizer, model

def greedy(model, tokenizer, input_ids, max_new_tokens=256):
    output_ids = model.generate(input_ids,
                                max_new_tokens=max_new_tokens,
                                do_sample=False)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def top_k(model, tokenizer, input_ids, max_new_tokens=256, temperature=0.7, top_k=50):
    
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

def beam_search_decoding(model, tokenizer, input_ids, max_new_tokens=256,  beam_size=3):
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        early_stopping=True,         
        do_sample=False,              
    )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def transition_fn(prob, beam_width):
    top_vals, top_indices = torch.topk(prob[0], k=beam_width)
    return [c.item() for c in top_indices], [c.item() for c in top_vals]

def score_fn(score):
    return np.log(score + 1e-8)

def extend_window(window_list, prob, window_size):
    window_list.append(prob)
    if len(window_list) > window_size:
        window_list.pop(0)
    return window_list

def my_decoding(model, tokenizer, input_ids, max_new_tokens=256, window_size=5, beam_width=3, alpha=0.5, stop_eos_count=1):
    
    window_list = []
    generated_tokens = input_ids.clone()
    # prefix_ids = input_ids.clone()
    beam = [(generated_tokens[-1], [], 0.0, [])]
    beam_iter = 0

    for i in range(max_new_tokens):

        print(f"token {i}")
        print("--"*10)
        
        with torch.no_grad():

            if len(window_list) == window_size:
                if beam_iter < window_size:
                    candidates = []
                    eos_count = 0

                    for _, seq, score, window in beam:
                        _input_ids = torch.cat([generated_tokens, torch.tensor(seq, device=generated_tokens.device).unsqueeze(0)], dim=-1) if seq else generated_tokens
                        _attention_mask = torch.ones_like(_input_ids)
                        
                        outputs = model(_input_ids, _attention_mask)  
                        logits = outputs.logits[:, -1, :]  
                        prob = F.softmax(logits, dim=-1)
                        
                        pmi = torch.log(prob/(window[0]+1e-8)) # pmi between y_window and y_t
                        ppmi_local = torch.max(pmi, torch.tensor(0.0, device=pmi.device))
                        ppmi_logits = logits + ppmi_local
                        pmi_prob = F.softmax(ppmi_logits, dim=-1)
                        
                        top_ids, top_scores = transition_fn(pmi_prob, beam_width)
                        for j, new_id in enumerate(top_ids):
                            if new_id == tokenizer.eos_token_id:
                                eos_count += 1
                            new_seq = seq + [new_id]
                            new_score = score + score_fn(top_scores[j])
                            new_window = extend_window(window, pmi_prob, window_size)
                            candidates.append((new_id, new_seq, new_score, new_window))
                        
                    if eos_count >= stop_eos_count:
                        final_seq = sorted(candidates, key=lambda x: x[2], reverse=True)[0][1]
                        generated_tokens = torch.cat([generated_tokens, torch.tensor(final_seq, device=generated_tokens.device).unsqueeze(0)], dim=-1)
                        break
                    
                    beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]			
                    beam_iter += 1
                    
                    print(f"**beam seq**")
                    for n in range(len(beam)):
                        # print(f"{beam[n][1]}    ", end="")
                        print(f"{tokenizer.convert_ids_to_tokens(beam[n][1])}   ", end="")
                    print()
                    
                else:
                    beam_seqs = [x[1] for x in beam]
                    beam_scores = [x[2] for x in beam]
                    beam_windows = [x[3] for x in beam]
                    prefix_scores = []
                    prefix_ids = generated_tokens[:, input_ids.shape[1]:]
                    
                    for seq in beam_seqs:
                        _input_ids = prefix_ids.clone()
                        _log_score = 0.0
                        for s, token in enumerate(seq):
                            _attention_mask = torch.ones_like(_input_ids)
                            outputs = model(_input_ids, _attention_mask)
                            logits = outputs.logits[:, -1, :]
                            prob = F.softmax(logits, dim=-1)
                            pmi = torch.log(prob/(window_list[s] + 1e-8)) 
                            ppmi_local = torch.max(pmi, torch.tensor(0.0, device=pmi.device))
                            ppmi_logits = logits + ppmi_local # prefix_pmi 계산할 때에도 pmi_local 적용한 확률 분포에서
                            pmi_prob = F.softmax(ppmi_logits, dim=-1)
                            
                            _log_score += score_fn(pmi_prob[0][token].item())
                            _input_ids = torch.cat([_input_ids, torch.tensor([token], device=_input_ids.device).unsqueeze(0)], dim=-1)
                        prefix_scores.append(_log_score)
                    
                    prefix_ppmi = [max(bs-ps, 0) for bs, ps in zip(beam_scores, prefix_scores)] # pmi between x and text chunck
                    rank_score = alpha * np.array(beam_scores) +  (1-alpha) * np.array(prefix_ppmi)
                    first_rank = np.argmax(rank_score)
                    
                    final_seq = beam_seqs[first_rank]
                    print(f"📌 final seq: {tokenizer.convert_ids_to_tokens(final_seq)}")
                    final_window = beam_windows[first_rank]
                    generated_tokens = torch.cat([generated_tokens, torch.tensor(final_seq, device=generated_tokens.device).unsqueeze(0)], dim=-1)
                    
                    # 재시작
                    beam = [(generated_tokens[-1], [], 0.0, final_window)]
                    window_list = final_window
                    beam_iter = 0
            else:
                candidates = []
                eos_count = 0
                
                for _, seq, score, window in beam:
                    _input_ids = torch.cat([generated_tokens, torch.tensor(seq, device=generated_tokens.device).unsqueeze(0)], dim=-1) if seq else generated_tokens
                    _attention_mask = torch.ones_like(_input_ids)
                    outputs = model(_input_ids, _attention_mask)
                    logits = outputs.logits[:, -1, :]
                    prob = F.softmax(logits, dim=-1)
                    
                    top_ids, top_scores = transition_fn(prob, beam_width)
                    for j, new_id in enumerate(top_ids):
                        if new_id == tokenizer.eos_token_id:
                            eos_count += 1
                        new_seq = seq + [new_id]
                        new_score = score + score_fn(top_scores[j])
                        new_window = window + [prob]
                        candidates.append((new_id, new_seq, new_score, new_window))
                    
                if eos_count >= stop_eos_count:
                    final_seq = sorted(candidates, key=lambda x: x[2], reverse=True)[0][1]
                    generated_tokens = torch.cat([generated_tokens, torch.tensor(final_seq, device=generated_tokens.device).unsqueeze(0)], dim=-1)
                    break
                
                beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
                print(f"**beam seq**")
                for n in range(len(beam)):
                    # print(f"{beam[n][1]}    ", end="")
                    print(f"{tokenizer.convert_ids_to_tokens(beam[n][1])}   ", end="")
                print()
                
                if len(beam) > 0 and len(beam[0][1]) == window_size:
                    _, max_path = max(enumerate(beam), key=lambda x: x[1][2])
                    max_generation = max_path[1]
                    max_window = max_path[3]
                    
                    generated_tokens = torch.cat([generated_tokens, torch.tensor(max_generation, device=generated_tokens.device).unsqueeze(0)], dim=-1)
                    window_list = max_window
                    beam = [(generated_tokens[-1], [], 0.0, window_list)]
                    beam_iter = 0
                    
                    print(f"📌 final_seq: {tokenizer.convert_ids_to_tokens(max_generation)}")
            
            if generated_tokens[0][-1].item() == tokenizer.eos_token_id:
                break
    
    return generated_tokens

def main():
    
    model_name = "facebook/opt-6.7b"
    article='''Acting president Rawhi Fattuh has announced today that Palestinian elections will be held on January 9. Futtuh, head of the Palestinian parliament, was sworn in hours after the death of Yasser Arafat on Thursday, and Palestinian Basic Law dictates that he may only serve up to two months before elections are held.'''
    prompt = f'''Write the continued text of the following article.\n\n{article}\n\n:Continued text: '''
    device = "cuda:1"
    tokenizer, model = load_model(model_name)
    model.to(device)
    
    prompt_input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = prompt_input["input_ids"].to(device)
    model.eval()
    
    greedy_output = greedy(model, tokenizer, input_ids)
    topk_output = top_k(model, tokenizer, input_ids)
    topp_output = top_p(model, tokenizer, input_ids)
    output_tokens = my_decoding(model,
                                tokenizer,
                                input_ids,
                                max_new_tokens=256,
                                window_size=10, # window 크기를 늘림
                                beam_width=3,
                                alpha=0.3) # alpha 조정 중요 (local에 더 힘을 줄 것이냐 prefix에 더 힘을 줄 것이냐를 결정) -> eos인데도 prefix pmi를 고려하다 보니까 continuation이 일어남...

    my_decoded_tokens = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    print(f"prompt: {prompt}\n\n")
    print("A:\n", greedy_output)
    print("__" * 50)
    print("B:\n", topk_output)
    print("__" * 50)
    print("C:\n", topp_output)
    print("__" * 50)
    print("D:\n", my_decoded_tokens)
    
if __name__ == "__main__":
    main()