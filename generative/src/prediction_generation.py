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
import yaml

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



class LLM:
    def __init__(self, config=None):
        self.data_path = config['data_path']
        self.train_data_path = self.data_path + config['train_data_file']
        self.test_data_path = self.data_path + config['test_data_file']

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        self.PROMPT_NO_QUESTION_PLUS = None
        self.PROMPT_QUESTION_PLUS = None

        model_path = config['model_path']
        output_dir = config['output_dir']

        # TODO 학습된 Checkpoint 경로 입력
        self.model_name = config['model_name'] #model_path + "outputs_gemma/checkpoint-4491"

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self._init_model()
        self._init_prompt()
        self._init_dataset()
    
    def _init_model(self):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

    def _init_dataset(self):
        # Load the test dataset
        # TODO Test Data 경로 입력
        test_df = pd.read_csv(self.test_data_path)

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
                )
            # <보기>가 없을 때
            else:
                user_message = self.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                )

            test_dataset.append(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": "당신은 수능 문제 출제자로 모든 문제의 답을 알고 있습니다. 지문을 읽고 질문의 답을 구하세요. 정답은 반드시 말해야 합니다."},
                        {"role": "user", "content": user_message},
                    ],
                    "label": row["answer"],
                    "len_choices": len_choices,
                }
            )
        
        self.test_dataset = test_dataset
    
    def _init_prompt(self):
        self.PROMPT_NO_QUESTION_PLUS = """지문:
        {paragraph}

        질문:
        {question}

        선택지:
        {choices}

        1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
        정답:"""

        self.PROMPT_QUESTION_PLUS = """지문:
        {paragraph}

        질문:
        {question}

        <보기>:
        {question_plus}

        선택지:
        {choices}

        1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
        정답:"""

        #self.set_prompt()

    def set_prompt(self, prompt_no_question_plus=None, prompt_question_plus=None):
        if prompt_no_question_plus:
            self.PROMPT_NO_QUESTION_PLUS = prompt_no_question_plus
        if prompt_question_plus:
            self.PROMPT_QUESTION_PLUS = prompt_question_plus
    
    def train(self):
        pass

    def evaluate(self):
        print(self.trainer.evaluate())

    def test(self):
        infer_results = []

        self.model.eval()
        with torch.inference_mode():
            for data in tqdm(self.test_dataset):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to("cuda")

                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                generated_text = self.tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
                infer_results.append({"id": _id, "answer": generated_text})

        # %%
        output = pd.DataFrame(infer_results)
        output.answer = output.answer.map(lambda x: 0 if x.strip() == "" else x)
        output.to_csv(self.data_path + "output.csv", index=False)

if __name__ == "__main__":
    config = None
    with open("../config/model.yaml") as f:
        config = yaml.safe_load(f)
    
    llm = LLM(config)
    llm.test()
        