from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from tqdm import tqdm
import os
import json
import torch
import argparse

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

def out_file(args, generation_lst):
    # dataset:contrastive_decoding 여부 (student/no):디코딩 종류 (no/contrastive_search_baseline/top-k/nucleus/typical/eta)
    output_path = os.path.join("/mnt/aix7101/minsuh/decoding_results", f"{args.model_type}_{args.prompt_file}_{args.contrastive_decoding}_{args.do_sample}_output_debug.jsonl")
    with open(output_path, 'w') as f:
        for kk in generation_lst:
            print(json.dumps(kk), file=f) 
    print(f'written to {output_path}')
    return 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--student_name_or_path",
        default=None,
        type=str
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
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")
    print(f"tokenizer is loaded.")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", device_map=config.device_map)
    
    datasets_val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    datasets_test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    dataset = concatenate_datasets([datasets_val, datasets_test])
    print(dataset)    
    
    column_names = dataset.column_names # wikitext: ['text'], wp: ['prompt', 'story'], wikinews: ['text']
    text_column_name = "text" if "text" in column_names else column_names[0] # wikitext: 'text', wp: 'prompt', wikinews: 'text'
    print(f"text_column_name: {text_column_name}")
    
    def tokenize_function(examples):
        examples[text_column_name] = [x.replace(' <newline>', '\n') for x in examples[text_column_name]]
        examples[text_column_name] = [tokenizer.bos_token + x for x in examples[text_column_name] if len(x) > 0]

        result_dict = tokenizer(examples[text_column_name], add_special_tokens=False) 
        input_ids_lst = [x[:32] for x in result_dict['input_ids'] if len(x) >= 160] # first 32 words as prompt 
        gold_lst = [x for x in result_dict['input_ids'] if len(x) >= 160] # add to golden list if length >= 160
        
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
    
    prompt_ids = tokenized_datasets[:2000]['input_ids'] 
    ref_lst = tokenized_datasets[:2000]['gold'] 
    ref_lst = tokenizer.batch_decode(ref_lst)
    ref_lst = [(0, x) for x in ref_lst]

    prompt_lst = tokenizer.batch_decode(prompt_ids)
    
    generation_lst = []
    
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
        num_return_sequences=args.num_return_sequences)
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