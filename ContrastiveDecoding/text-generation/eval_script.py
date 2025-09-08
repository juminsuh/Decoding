# parse the generated results into a list of text
from dis import disco
import json, sys, os 
import numpy as np 
our_file = sys.argv[1] # 첫 번째 인자를 our_file로 사용

rewrite=False  
mauve = True 
coherence = True 
ppl=True 
entity_f1 = True 
disco_coh= False  
 

if rewrite: # output.jsonl 파일의 gen_text를 gold_ref로 덮어씀
    with open(our_file, 'r') as f1,  open(our_file[:-6]+'_gold.jsonl', 'w') as f2:
        examples = [json.loads(l.strip()) for l in f1]
        for x in examples:
            x[0]['gen_text'] = x[0]['gold_ref']
            print(json.dumps(x), file=f2)
    our_file = our_file[:-6]+'_gold.jsonl'

print(our_file)
cumulative_stats = {}
# sys.path.insert(0, '/private/home/xlisali/decoding/text-generation/baselines/SimCTG')

def load_file_ref_ours(our_file, cont_only=True):
    text_list = []
    text_ref_lst = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
            text_list.append(text)
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            i=0
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_ref = example['gold_ref']
                    if cont_only: # prompt 이후 continuation만 추출
                        text_prompt = example['prompt']
                        if text_prompt.startswith('</s>'):
                            text_prompt = text_prompt.lstrip('</s>')
                        if text_ref.startswith('</s>'):
                            text_ref = text_ref.lstrip('</s>')
                        if text.startswith('</s>'):
                            text = text.lstrip('</s>')
                        try:
                            assert text_prompt == text[:len(text_prompt)]
                            # assert text_prompt == text_ref[:len(text_prompt)]
                        except:
                            continue
                        text = text[len(text_prompt):]
                        if "wikitext" in our_file or "wikinews" in our_file:
                            text_ref = text_ref[len(text_prompt):]
                    text_list.append(text)
                    text_ref_lst.append(text_ref)
                else:
                    assert False, 'invalid formatting'
    return text_ref_lst, text_list # gold_ref, gen_text


def load_file_(our_file):
    text_list = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
        #     text = item['generated_result']['0']['full_text']
            text_list.append(text)
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_prefix = example['prompt']
                    if len(text_prefix) >= len(text): # gen_text 길이가 prompt보다 짧으면 text_list에 포함 x
                        continue 
                    text_list.append(text[len(text_prefix):])
                else:
                    text = example['trunc_gen_text']
                    text_list.append(text)
    return text_list # generation만 포함하는 리스트

def load_file_pair(our_file):
    text_list = []
    if our_file.endswith('json'):
        with open(our_file) as f:
            item_list = json.load(f)
        for item in item_list:
            text = item['generated_result']['0']['continuation']
        #     text = item['generated_result']['0']['full_text']
            prefix = item['prefix_text']
            text_list.append((prefix, text))
    else: # for jsonl files. 
        with open(our_file, "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            for example in examples:
                if isinstance(example, list):
                    example = example[0]
                if 'gen_text' in example:
                    text = example['gen_text']
                    text_prefix = example['prompt']
                    if len(text_prefix) >= len(text): # gen_text 길이가 prompt보다 짧으면 text_list에 포함 x
                        continue 
                    text_list.append((text_prefix, text[len(text_prefix):]))
                else:
                    text = example['trunc_gen_text']
                    try:
                        text_prefix = example['prompt']
                    except:
                        text_prefix = example['prefix']
                    text_list.append((text_prefix,text))
    return text_list

def process_text(text_lst):
    for i, text in enumerate(text_lst):
        temp_sent = text.replace(' @', '').replace('@ ', '') # remove space 
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        temp_sent = TreebankWordDetokenizer().detokenize(temp_sent.split())
        # print(temp_sent) 
        text_lst[i] = temp_sent
    return text_lst 

text_list = load_file_(our_file)
# compute the evaluation results
from simctg.evaluation import measure_repetition_and_diversity
rep_2, rep_3, rep_4, diversity = measure_repetition_and_diversity(text_list) # generation의 rep-n, diversity 
print (f'rep-2 is {rep_2}, rep-3 is {rep_3}, rep-4 is {rep_4}, and diversity is {round(diversity, 2)}')
cumulative_stats['name'] = our_file
cumulative_stats['rep-2'] = rep_2
cumulative_stats['rep-3'] = rep_3
cumulative_stats['rep-4'] = rep_4
cumulative_stats['diversity'] = diversity #round(diversity,2)

'''
   The result of rep-2 is 3.93, rep-3 is 0.78, rep-4 is 0.31, and diversity is 0.95
'''
if mauve:

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    p_text, q_text = load_file_ref_ours(our_file, cont_only=True) # gold_ref, gen_text

    print(len(p_text), len(q_text))
    # tokenize by GPT2 first. 
    tgt_len = 128
    x = tokenizer(p_text, truncation=True, max_length=tgt_len)['input_ids']
    y = tokenizer(q_text, truncation=True, max_length=tgt_len)['input_ids']
    # for fair distribution comparison
    xxyy = [(xx, yy) for (xx, yy) in zip(x, y) if len(xx) == tgt_len and len(yy) == tgt_len]
    x, y = zip(*xxyy)
    # map back to texts. 
    p_text = tokenizer.batch_decode(x)#[:target_num]
    q_text = tokenizer.batch_decode(y)#[:target_num]
    print(len(p_text), len(q_text))
    
    import mauve 
    ref_list = p_text
    pred_list = q_text 
    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
    out = mauve.compute_mauve(p_text=ref_list, q_text=pred_list, device_id=1, max_text_length=256, 
        verbose=False, featurize_model_name='gpt2')
    # print(out)
    print(f"mauve: {out.mauve}") # prints 0.9917, 
    cumulative_stats['mauve'] = out.mauve
    

if coherence:
    print('Evaluating coherence score')

    from simcse import SimCSE
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    
    sent_lst = load_file_pair(our_file)
    full_sim_lst = []
    pp_lst, yy_lst = zip(*sent_lst) # prompt, gen_text
    pp_lst = list(pp_lst)
    yy_lst = list(yy_lst) 
    print(len(pp_lst), len(yy_lst))
        
    similarities = model.similarity(pp_lst, yy_lst, device="cuda:1")
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 
    cumulative_stats['coherence'] = coherence_score
    print(f"coherence: {round(coherence_score, 2)}")

# print(cumulative_stats)
# str_head = ''
# str_ = ''
# for k, v in cumulative_stats.items():
#     str_head += f"\t{k}&"
#     if isinstance(v, float):
#         str_ += f'\t{round(v, 2)}&'
#     else:
#         str_ += f'\t{v}&'
# print(str_head + '\\\\')
# print(str_ + '\\\\') 