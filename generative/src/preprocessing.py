# %%
import torch
import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig
import bitsandbytes as bnb
from datasets import Dataset
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
    PeftModel
)

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


import os
import yaml

from huggingface_hub import login

from utils import *

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

def tokenize_dataset(tokenizer, dataset):
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=False,#True,
        desc="Tokenizing",
    )

    return tokenized_dataset

class DataProcessor:
    def __init__(self, config=None):
        self.config = config

        self.data_path = None
        self.train_data_path = None
        self.test_data_path = None

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        self.PROMPT_NO_QUESTION_PLUS = None
        self.PROMPT_QUESTION_PLUS = None

        self.system_message = None

        self.max_new_tokens = None

        self.model_path = None
        self.output_dir = None

        self.model_name = None
        self.device_map = None
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self.trainer = None

        # metric 로드
        self.acc_metric = evaluate.load("accuracy")
        # 정답 토큰 매핑
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    def init(self):
        self._init_config()
        self._init_prompt()
        self._init_dataset()

    def _init_config(self):
        config = self.config
        self.data_path = config['data_path']
        self.train_data_path = self.data_path + config['train_data_file']
        self.test_data_path = self.data_path + config['test_data_file']

        self.max_new_tokens = config['max_new_tokens']

        self.model_path = config['model_path']
        self.output_dir = config['output_dir']

        self.model_name = config['model_name']
        self.device_map = config['device_map']

        self.batch_size = config['batch_size']
        self.num_train_epochs = config['num_train_epochs']
        self.max_seq_length = config['max_seq_length']

    def _get_records(self, data_path):
        df = pd.read_csv(data_path) 

        # Flatten the JSON dataset
        records = []
        for _, row in df.iterrows():
            problems = literal_eval(row['problems'])

            if self.config['choices_shuffle'] and problems.get('answer', None):
                answer = problems.get('answer', None)
                answer_text = problems['choices'][answer-1]

                random.shuffle(problems['choices'])
                problems['answer'] = problems['choices'].index(answer_text)+1

            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
                "hint": row.get('hint', None),
                "class": row.get('class', None)
            }

            records.append(record)
        
        return records

    # %%
    def _formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def _tokenize(self, element):
        outputs = self.tokenizer(
            self._formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    def _init_dataset(self):
        self._init_train_dataset()
        self._init_test_dataset()

    def _init_train_dataset(self):
        records = self._get_records(self.train_data_path)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(df)

        processed_dataset = []
        for i in range(len(dataset)):
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = self.PROMPT_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                    hint=dataset[i]["hint"],
                )
            # <보기>가 없을 때
            else:
                user_message = self.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    choices=choices_string,
                    hint=dataset[i]["hint"],
                )

            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": dataset[i]["id"],
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                    ],
                    "label": dataset[i]["answer"],
                    "class": dataset[i]["class"],
                }
            )

        #print(processed_dataset)
        processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
        self.train_dataset = processed_dataset

    def _init_test_dataset(self):
        records = self._get_records(self.test_data_path)
                
        # Convert to DataFrame
        test_df = pd.DataFrame(records)
        
        test_dataset = []
        for i, row in test_df.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            len_choices = len(row["choices"])
            
            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = self.PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                    hint=row["hint"],
                )
            # <보기>가 없을 때
            else:
                user_message = self.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                    hint=row["hint"],
                )

            test_dataset.append(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_message},
                    ],
                    "label": row["answer"],
                    "len_choices": len_choices,
                    "class": row["class"],
                }
            )
        
        self.test_dataset = test_dataset
    
    def _init_prompt(self):
        self.PROMPT_NO_QUESTION_PLUS = """지문:
        {paragraph}

        질문:
        {question}

        힌트:
        {hint}

        선택지:
        {choices}

        1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
        정답:"""

        self.PROMPT_QUESTION_PLUS = """지문:
        {paragraph}

        질문:
        {question}

        힌트:
        {hint}

        <보기>:
        {question_plus}

        선택지:
        {choices}

        1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
        정답:"""

        self.system_message = "당신은 수능 문제 출제자로 모든 문제의 답을 알고 있습니다. 지문을 읽고 질문의 답을 구하세요. 정답은 반드시 말해야 합니다."

        #self.set_prompt()

    def set_prompt(self, prompt_no_question_plus=None, prompt_question_plus=None):
        if prompt_no_question_plus:
            self.PROMPT_NO_QUESTION_PLUS = prompt_no_question_plus
        if prompt_question_plus:
            self.PROMPT_QUESTION_PLUS = prompt_question_plus

        self._init_dataset()
        
    def set_user_prompt(self, prompt_no_question_plus=None, prompt_question_plus=None):
        if prompt_no_question_plus:
            self.PROMPT_NO_QUESTION_PLUS = prompt_no_question_plus
        if prompt_question_plus:
            self.PROMPT_QUESTION_PLUS = prompt_question_plus

        self._init_dataset()

    def set_system_prompt(self, system_message=None):
        if system_message:
            self.system_message = system_message

        self._init_dataset()
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset