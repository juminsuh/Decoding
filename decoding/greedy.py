import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
import config
import os

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None) # float16으로 하면 Underflow돼서 nan값이 생겨서 에러남
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=None)
    if model_name == 'facebook/opt-6.7b':
        dispatch_model(model, device_map=config.device_map)
    if model_name == 'openai-community/gpt2-xl':
        model.to(device)
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

def greedy_with_entropy(model, tokenizer, input, max_new_tokens=256):
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']

    # 복사본 생성
    generated = input_ids.clone()
    mask = attention_mask.clone()

    entropies = []
    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=mask)
            logits = outputs.logits[:, -1, :]  # 마지막 토큰의 logits
            probs = F.softmax(logits, dim=-1)

            # Entropy 계산: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # shape: (batch_size,)
            entropies.append(entropy.item())

            # greedy하게 다음 토큰 선택
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)  # shape: (batch_size, 1)

            # 다음 토큰 추가
            generated = torch.cat([generated, next_token], dim=-1)
            new_mask = torch.ones_like(next_token)
            mask = torch.cat([mask, new_mask], dim=-1)

            # 종료 조건 (e.g. EOS 토큰)
            if next_token.item() == tokenizer.eos_token_id:
                break

    decoded = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return decoded, entropies

def sampling_with_entropy(model, tokenizer, input, max_new_tokens=256, top_k=10):
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']

    # 복사본 생성
    generated = input_ids.clone()
    mask = attention_mask.clone()

    entropies = []
    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=mask)
            logits = outputs.logits[:, -1, :]  # 마지막 토큰의 logits
            probs = F.softmax(logits, dim=-1)

            # Entropy 계산: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # shape: (batch_size,)
            entropies.append(entropy.item())

            # top-k filtering
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)  # shape: (batch_size, top_k)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # 정규화

            # Top-k 중에서 샘플링
            next_token = torch.multinomial(topk_probs, num_samples=1)  # shape: (batch_size, 1)
            next_token_id = topk_indices.gather(-1, next_token)  # 원래 vocab 기준의 ID로 변환

            # 다음 토큰 추가
            generated = torch.cat([generated, next_token_id], dim=-1)
            new_mask = torch.ones_like(next_token_id)
            mask = torch.cat([mask, new_mask], dim=-1)

            # 종료 조건
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    decoded = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return decoded, entropies

import matplotlib.pyplot as plt
import numpy as np

def plot_entropy(entropies, type):
    """
    entropies: list of float — conditional entropy per decoding step
    save_path: str or None — if given, the plot will be saved to this path
    """
    save_dir = "./img"
    save_path = os.path.join(save_dir, f"entropies-{type}.png")
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

def main():
    model_name = "facebook/opt-6.7b"
    device="cuda:0"
    tokenizer, model = load_model(model_name, device=device)
    
    prompt='''Acting president Rawhi Fattuh has announced today that Palestinian elections will be held on January 9. Futtuh, head of the Palestinian parliament, was sworn in hours after the death of Yasser Arafat on Thursday, and Palestinian Basic Law dictates that he may only serve up to two months before elections are held.'''
    input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    texts, entropies = greedy_with_entropy(model=model, tokenizer=tokenizer, input=input)
    texts2, entropies2 = sampling_with_entropy(model=model, tokenizer=tokenizer, input=input, top_k=10)
    
    print(f"greedy decoded text: {texts}\n\n")
    print(f"sampling decoded text: {texts2}\n\n")
    plot_entropy(entropies=entropies, type="greedy")
    plot_entropy(entropies=entropies2, type="sampling")
    
if __name__=="__main__":
    main()