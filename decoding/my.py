import os
import numpy as np
import scipy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42)

model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None) # float16으로 하면 Underflow돼서 nan값이 생겨서 에러남
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=None)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

model.to(device)

def plot_prob(tensor, t, name):

    save_dir = "./img"
    save_path = os.path.join(save_dir, f"prob_dist_{name}_{t}.png")
    plt.figure(figsize=(12, 4))
    y = tensor.to("cpu").numpy().squeeze()
    plt.plot(range(len(tensor[0])), y)
    plt.xlabel("Vocab Index")
    plt.ylabel("Probability")
    plt.title(f"Probability distribution of {t}-th token")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def jsd(p, q, base=np.e, eps=1e-12):
    p = p.to("cpu").numpy().astype(np.float32)
    q = q.to("cpu").numpy().astype(np.float32)

    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    jsd_val = 0.5 * sp.stats.entropy(p, m, base=base) + 0.5 * sp.stats.entropy(q, m, base=base)
    return jsd_val

def kld(p, q, eps=1e-8):
    
    p = p.to("cpu").numpy().astype(np.float32)
    q = q.to("cpu").numpy().astype(np.float32)
    
    p += eps
    q += eps
    p = p / p.sum()
    q = q / q.sum()
    
    # epsilon 추가로 안정성 확보
    # p = torch.clamp(p, min=eps)
    # q = torch.clamp(q, min=eps)

    kl = (p * (np.log(p) - np.log(q))).sum()
    return kl

def _top_k_sampling(logits: torch.Tensor, top_k=20,
                    filter_value=-float("Inf"),
                    min_tokens_to_keep=1) -> torch.Tensor:
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]  # top-k보다 작은 값 제거
    logits[indices_to_remove] = filter_value
    top_k_prob = F.softmax(logits, dim=-1)
    return top_k_prob

def standard_decoding(input_ids, max_length=256, temperature=0.5, top_k=20, top_p=0.9):

	output_ids = model.generate(input_ids, 
								max_length=max_length,
								temperature=temperature,
								top_k=top_k,
								top_p=top_p,
								do_sample=True)

	return tokenizer.decode(output_ids[0], skip_special_tokens=True)

prompt = '''What is the hopeful result of going to see a play? A. being entertained, B. meet, C. sit. Choose only one answer.'''

prompt_input = tokenizer(prompt, return_tensors="pt")
input_ids = prompt_input["input_ids"].to(device)
# attention_mask = prompt_input["attention_mask"].to(device)

def my_decoding(model, tokenizer, input_ids, max_length, window_size, epsilon, alpha, temperature=0.5):
    previous_list = []
    window_list = []
    generated_tokens = input_ids.clone()

    for i in range(2):

        print(f"token {i}")
        print("--"*10)
        attention_mask = torch.ones_like(generated_tokens)
        print(f"attention_mask_length: {len(attention_mask[0])}\n")
        with torch.no_grad():

            # tip: float16으로 바꾸기 -> 연산량 줄어듦
            # 1. 원래 분포 생성
            outputs = model(generated_tokens, attention_mask)  # logit만 계산 <-> .generate()는 auto-regressive하게 계산
            logits = outputs.logits[:, -1, :]  # (batch_size, seq_len, vocab_size) -> 마지막 토큰의 로짓값
            prob = F.softmax(logits / temperature, dim=-1)
            
            # if i == 0:
            #     prefix_prob = prob # 맨 처음 prob disrtibution으로 고정
            # max_prob = prob.max(dim=-1, keepdim=True).values
            # threshold = max_prob * 0.1
            # filtered_prob = torch.where(prob >= threshold, prob, torch.zeros_like(prob))

            original_token = torch.multinomial(prob, num_samples=1)
            print(f"original prob: {prob[0, original_token].item()}")
            print(f"original token: {original_token.item()}")
            print(f"original decode: {tokenizer.convert_ids_to_tokens(original_token.item())}\n")
            # print(f"prob: {prob.sum()}")
            
            # 1. contrastive
            # if previous_list:
            #     max_diff = 0
            #     max_diff_idx = 0
            #     for j, previous_prob in enumerate(previous_list):
            #         # print(f"window prob: {window_prob.sum()}")
            #         diff = kld(prob[0], previous_prob[0])
            #         # print(f"diff {w}: {diff:.8f}")
            #         if diff > max_diff:
            #             max_diff = diff
            #             max_diff_idx = j  # JSD가 가장 큰 window index
            #     print(f"max diff: {max_diff}")
            #     print(f"max diff j: {max_diff_idx}\n")
            #     # 2-2. contrast prob and window_prob
            #     max_diff_prob = previous_list[max_diff_idx]

            #     # plot_prob(prob, i, "prob")
            #     # plot_prob(max_diff_prob, max_diff_idx, "max_diff_prob")
            #     contrastive_prob = torch.log(prob / (max_diff_prob+1e-8))  # p'
                # contrastive_prob = F.softmax(torch.log(prob / (max_diff_prob+1e-8)), dim=-1)  # p'
                # plot_prob(contrastive_prob, i, "contrastive_prob")
            # else:
            #     contrastive_prob = prob

            # 2. pmi
            if len(window_list) == window_size: # window가 형성되면

                # contrastive_token = torch.multinomial(contrastive_prob, num_samples=1)
                pmi = torch.log(prob/(window_list[0]+1e-8))
                coherence_score = torch.max(pmi, torch.tensor(0.0, device=pmi.device))
                top_cohs, top_indices = torch.topk(coherence_score[0], k=5)
                print(f"top_cohs: {top_cohs}")
                print(f"top_indices: {top_indices}")
                print(f"<top_decodes>")
                for idx in top_indices:
                    print(f"{tokenizer.convert_ids_to_tokens(idx.item())}   ", end="")
                print('\n')
                # print(f"coherence_score: {coherence_score}\n")
                
                # prefix_pmi = torch.log2(prob/prefix_prob+1e-8)
                # prefix_coherence_score = torch.max(prefix_pmi, torch.tensor(epsilon, device=pmi.device))
                # print(f"prefix_coherence_score = {prefix_coherence_score}")

                adjusted_logit = logits + coherence_score
                adjusted_prob_store = F.softmax(adjusted_logit, dim=-1)
                adjusted_prob = _top_k_sampling(adjusted_logit)
                # print(f"k: {len(adjusted_logit_k[0])}") # 256000
                # repetition_penalty = -torch.log(contrastive_prob) / torch.sum(-torch.log(contrastive_prob), dim=-1)
                # plot_prob(adjusted_prob, i, "adjusted_prob")
            else:  # window가 형성되지 않았을 때: 일반 top-k decoding
                adjusted_prob_store = prob
                adjusted_prob = _top_k_sampling(logits)

            # 4. next token generation
            print(f"adjusted_prob_sum: {adjusted_prob.sum()}")
            next_token = torch.multinomial(adjusted_prob, num_samples=1)
            # next_token = torch.argmax(contrastive_prob, dim=-1, keepdim=True)
            # print(f"original contrastive prob: {adjusted_prob[0, original_token].item()}\n") # adjusted_prob에 original_token이 없으면 문제 발생??
            print(f"adjusted prob: {adjusted_prob[0, next_token].item()}")
            print(f"adjusted decode: {tokenizer.convert_ids_to_tokens(next_token.item())}\n")
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # print(f"next_token.item: {next_token.item()}")
            # print(f"decode: {tokenizer.convert_ids_to_tokens(next_token.item())}")

            if next_token.item() == tokenizer.eos_token_id:
                break
            
            previous_list.append(adjusted_prob_store)
            # 5. window update
            window_list.append(adjusted_prob_store)
            if len(window_list) > window_size:
                window_list.pop(0)
            print(f"window_size: {len(window_list)}")

    return generated_tokens

def main():
    model.eval()
    standard_output = standard_decoding(input_ids)
    output_tokens = my_decoding(
                                model,
                                tokenizer,
                                input_ids,
                                max_length=256,
                                window_size=5,
                                epsilon=0.5,
                                alpha = 0.5,
                                temperature=0.5
                                )

    my_decoded_tokens = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    print("Standard Decoding Output:\n", standard_output)
    print("__" * 50)
    print("My Decoding Output:\n", my_decoded_tokens)
    
if __name__ == "__main__":
    main()