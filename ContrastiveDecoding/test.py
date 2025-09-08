from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer
    # LlamaForCausalLM,
    # LlamaTokenizer
)
# import config

# model_name="gpt2-xl"
# # model_name="gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# print("model is successfully loaded.")
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model.to("cuda:0")
# # # print(len(model.model.decoder.layers))
# # print(model.generation_config)
model_name="meta-llama/Llama-2-7b-hf"
# # model_name="facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.to("cuda:0")
print("model is loaded successfully.")
# # print(len(model.model.decoder.layers))
# # help(model.generate)

# student_lm = OPTForCausalLM.from_pretrained("facebook/opt-125m").to("cuda:0")
# print("student model is loaded.")

prompt = "I want to go to Austria."
input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
input_ids = input_ids.to("cuda:0")

try:
    # output_sequences = model.generate(
    # input_ids=input_ids,
    # max_length=256,
    # min_length=256,
    # temperature=1.0,
    # top_k=0,
    # top_p=1.0,
    # eta_cutoff=0.0003,
    # repetition_penalty=1.0,
    # do_sample=True,
    # num_beams=1,
    # num_return_sequences=1,
    # student_lm=student_lm,
    # teacher_student=False,
    # model_kwargs_student={}, 
    # st_coef=1.0)
    # print('eta sampling')
    # print("✅ eta_cutoff 인자 사용 가능!")
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=256,
        min_length=256,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        # min_prob=0.0,
        repetition_penalty=1.0,
        do_sample=True,
        num_beams=1,
        num_return_sequences=1,
        # student_lm=student_lm,
        # teacher_student=True,
        # model_kwargs_student={}, 
        # st_coef=1.0,
        # tokenizer=tokenizer, # analysis
        # student_min_prob=0.0,
        # student_temperature=0.5,
        # use_cap_student='no', #cap student debug
        # use_switch='no'
    )
except TypeError as e:
    print("❌ 에러:", e)
    
output = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)
print(f"output text: {output}")

# from simctg.evaluation import measure_repetition_and_diversity
# from simcse import SimCSE
# model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

# from datasets import load_dataset
# datasets = load_dataset("wikitext", 'wikitext-103-raw-v1', split="validation")
# print(datasets)
# print(datasets['text'])

# import os
# import pandas as pd
# import json
# from datasets import Dataset, load_dataset, load_from_disk
# import datasets

# our_file = "/mnt/aix7101/minsuh/decoding_results/gpt2_wikitext_student_no_output.jsonl"
# our_file = "/mnt/aix7101/minsuh/decoding_results/gpt2_wp_student_no_output.jsonl"
# i=0
# with open(our_file, "r") as fin:
#     examples = [json.loads(l.strip()) for l in fin]
#     for example in examples:
#         if isinstance(example, list):
#             example = example[0]
#             print(f"example: {example}")
#         if i < 3:
#             break
#         i+=1
    
    

# file_path = "data/multilingual_wikinews.jsonl"
# save_path = "data/wikinews_en"

# def save_dataset():
#     if not os.path.exists(save_path):
#         texts = []
#         with open(file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 obj = json.loads(line)  
#                 if obj.get("lang") == "en":
#                     texts.append(obj.get("text", ""))

#         dataset = Dataset.from_dict({"text": texts})
#         dataset.save_to_disk(save_path)
#     return save_path

# dataset = datasets.load_from_disk(save_dataset())

# dataset = load_dataset("euclaise/writingprompts", split="test")
# # print(dataset)
# i=0
# for example in dataset:
#     print(f"example: {example}")  
#     if i >= 3:
#         break
#     i += 1


                
            



