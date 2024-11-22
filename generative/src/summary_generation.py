# %%
import torch
import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig

import os
from huggingface_hub import login

pd.set_option('display.max_columns', None)

with open("../.env", "r") as f:
    token = f.read().strip() 
os.environ['HUGGINGFACE_TOKEN'] = token

# %%
# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)

# %% [markdown]
# ### Load Data

# %%
data_path = '../data/'
train_data_path = data_path + "train.csv"
test_data_path = data_path + "test.csv"

model_path = '../model/'
output_dir = '../model/'
# %% [markdown]
# ## Inference

# %%
# TODO 학습된 Checkpoint 경로 입력
checkpoint_path = "maywell/EXAONE-3.0-7.8B-Instruct-Llamafied" # "Qwen/Qwen2.5-7B-Instruct" #model_path + "outputs_gemma_base/checkpoint-4491"
#checkpoint_path = model_path + "outputs_gemma_base/checkpoint-4491"

print(checkpoint_path)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

#model = AutoPeftModelForCausalLM.from_pretrained(
#    checkpoint_path,
#    trust_remote_code=True,
#    # torch_dtype=torch.bfloat16,
#    device_map="auto",
#)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
)

# %%
# Load the test dataset
# TODO Test Data 경로 입력
test_df = pd.read_csv(test_data_path)

# Flatten the JSON dataset
records = []
for _, row in test_df.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        "question_plus": problems.get('question_plus', None),
    }
    # Include 'question_plus' if it exists
    if 'question_plus' in problems:
        record['question_plus'] = problems['question_plus']
    records.append(record)
        
# Convert to DataFrame
test_df = pd.DataFrame(records)

# %%
PROMPT_NO_QUESTION_PLUS = """
지문과 질문을 읽고 학생들에게 도움이 될 수 있는 힌트를 제공합니다.
힌트는 500자 이내의 한국어로 작성합니다.

지문:
{paragraph}

질문:
{question}

힌트:"""

PROMPT_QUESTION_PLUS = """
지문과 질문을 읽고 학생들에게 도움이 될 수 있는 힌트를 제공합니다.
힌트는 500자 이내의 한국어로 작성합니다.

지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

힌트:"""

# %%

tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"

test_dataset = []
for i, row in test_df.iterrows():
    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
    len_choices = len(row["choices"])
    
    # <보기>가 있을 때
    if row["question_plus"]:
        user_message = PROMPT_QUESTION_PLUS.format(
            paragraph=row["paragraph"],
            question=row["question"],
            question_plus=row["question_plus"],
            choices=choices_string,
        )
    # <보기>가 없을 때
    else:
        user_message = PROMPT_NO_QUESTION_PLUS.format(
            paragraph=row["paragraph"],
            question=row["question"],
            choices=choices_string,
        )

    test_dataset.append(
        {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "대한민국 수능 전문가이다. 수능 국어 지문과 질문이 주어졌을 때, 학생들이 잘 풀 수 있도록 힌트를 제공하고자 한다."},
                {"role": "user", "content": user_message},
            ],
            "label": row["answer"],
            "len_choices": len_choices,
        }
    )

# %%time

infer_results = []

c=0
model.eval()
with torch.inference_mode():
    for data in tqdm(test_dataset):
        _id = data["id"]
        messages = data["messages"]
        len_choices = data["len_choices"]

        inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")

        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()

        infer_results.append({"id": _id, "answer": generated_text})

        c+=1

        if c==5:break

print(infer_results)

# %%
#output = pd.DataFrame(infer_results)

# %%
#output.answer = output.answer.map(lambda x: 0 if x.strip() == "" else x)

# %%
#output.to_csv(data_path + "output.csv", index=False)


