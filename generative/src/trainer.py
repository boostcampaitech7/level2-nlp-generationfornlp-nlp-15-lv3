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
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


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

def get_model(config):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 또는 load_in_4bit=True로 설정
        #llm_int8_threshold=6.0,  # 필요한 경우 추가 설정
        #llm_int8_skip_module="module_name",  # 옵션: 특정 모듈 스킵
    )

    #quantization_config = bnb.nn.LinearNF4()
    
    # NF4 양자화를 위한 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, # 모델을 4비트 정밀도로 로드
        bnb_4bit_quant_type="nf4", # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
        bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
        bnb_4bit_compute_dtype=torch.bfloat16 # 연산 속도를 높이기 위해 사용 (default: torch.float32)
    )
    
    if config['use_gptq']:
        """
        # 모델 로드 (기본 모델을 훈련된 GPT 모델로 대체)
        quantize_config = BaseQuantizeConfig(bits=4, group_size=128)

        model = AutoGPTQForCausalLM.from_pretrained(config['model_name'], quantize_config=quantize_config)
        """
        model_id = config['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if config['use_fp16'] else torch.float32,
            device_map=config['device_map'], 
            quantization_config=quantization_config)
        print("AutoGPTQForCausalLM")
    else:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                config["model_name"],
                trust_remote_code=True,
                torch_dtype=torch.float16 if config['use_fp16'] else torch.float32,
                device_map=config["device_map"],
                quantization_config=quantization_config if config['use_quant'] else None,
                #load_in_4bit=True,
            )
            print("AutoPeftModelForCausalLM")
        except:
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                trust_remote_code=True,
                torch_dtype=torch.float16 if config['use_fp16'] else torch.float32,
                device_map=config["device_map"],
                quantization_config=quantization_config if config['use_quant'] else None,
                #load_in_4bit=True,
            )
            print("AutoModelForCausalLM")
    
    return model

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
    )

    # pad token 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.special_tokens_map
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"

    return tokenizer

def get_collator(config, tokenizer):
    # Completion 부분만 학습하기 위한 data collator 설정
    # - 텍스트 중 response_template 까지는 ignore_index 로 loss 계산에서 제외
    # - 텍스트 중 response_template 이후는 학습에 포함 (정답 + eos 토큰)
    response_template = "<start_of_turn>model"

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    return data_collator
    
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
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
                "hint": row.get('hint', None),
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
    


class CustomSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs):
        print(f"Training step inputs: {inputs['input_ids'].shape}")
        return super().training_step(model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        print(f"Eval step inputs: {inputs['input_ids'].shape}")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
class LLM:
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
        self.lora_config = None
        self.sft_config = None
        self.trainer = None

        # metric 로드
        self.acc_metric = evaluate.load("accuracy")
        # 정답 토큰 매핑
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


    
    def init(self):
        self._init_config()
        #self._init_model()
        #self._init_collator()
        #self._init_prompt()
        #self._init_dataset()
        self._init_trainer()

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
        self.learning_rate = config['learning_rate']
        self.max_seq_length = config['max_seq_length']
        self.use_lora = config['use_lora']

    def set_model(self, model):
        self.model = model
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def set_collator(self, collator):
        self.data_collator = collator

    def set_train_dataset(self, dataset):
        self.train_dataset = dataset
        self._init_train_dataset()
    
    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def _init_model(self):
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self.config["device_map"],
            )
            print("AutoPeftModelForCausalLM")
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self.config["device_map"],
            )
            print("AutoModelForCausalLM")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        tokenizer = self.tokenizer
        # pad token 설정
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.special_tokens_map
        tokenizer.padding_side = 'right'
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"


    def _init_collator(self):
        # Completion 부분만 학습하기 위한 data collator 설정
        # - 텍스트 중 response_template 까지는 ignore_index 로 loss 계산에서 제외
        # - 텍스트 중 response_template 이후는 학습에 포함 (정답 + eos 토큰)
        response_template = "<start_of_turn>model"

        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

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
    
    def _init_train_dataset(self):
        # vram memory 제약으로 인해 인풋 데이터의 길이가 1024 초과인 데이터는 제외하였습니다. *힌트: 1024보다 길이가 더 긴 데이터를 포함하면 더 높은 점수를 달성할 수 있을 것 같습니다!
        #tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 1200) #max_seq_length) #1024)  
        dataset = self.train_dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = dataset['train']
        self.eval_dataset = dataset['test']
    
    def _init_test_dataset(self):
        pass


    # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def _preprocess_logits_for_metrics(self, logits, labels):
        tokenizer = self.tokenizer

        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx] # -2: answer token, -1: eos token
        return logits

    # metric 계산 함수
    def _compute_metrics(self, evaluation_result):
        tokenizer = self.tokenizer
        logits, labels = evaluation_result

        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: self.int_output_map[x], labels))

        # 소프트맥스 함수를 사용하여 로그트 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits).float(), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 정확도 계산
        acc = self.acc_metric.compute(predictions=predictions, references=labels)
        return acc
    
    def _init_trainer(self):
        self.lora_config = LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.sft_config = SFTConfig(
            fp16=self.config['use_mx_fp16'],
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine",
            max_seq_length=self.max_seq_length,
            output_dir=self.output_dir+self.model_name,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            save_only_model=True,
            report_to="none",
            #remove_unused_columns=False,
        )
        print(self.sft_config)

        if self.use_lora:
            # 양자화된 모델을 학습하기 전, 전처리를 위해 호출
            self.model = prepare_model_for_kbit_training(self.model)
            # LoRA 학습을 위해서는 아래와 같이 peft를 사용하여 모델을 wrapping 해주어야 함
            self.model = get_peft_model(self.model, self.lora_config)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
            peft_config=self.lora_config if self.use_lora else None,
            args=self.sft_config,
        )


    def train(self):
        self.trainer.train()
        self.evaluate()

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
                    #min_length=1,
                    #temperature=0.7,
                    #top_p=0.8
                )

                generated_text = self.tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
                infer_results.append({"id": _id, "answer": remove_not_digital(remove_not_numeric(generated_text))})

                #print(infer_results)

        # %%
        output = pd.DataFrame(infer_results)
        print(output)
        output.answer = output.answer.map(lambda x: 0 if x.strip() == "" else x)
        output.to_csv(self.data_path + "output.csv", index=False)

    def generate(self, data_path=None, dataset=None):
        model = self.model
        tokenizer = self.tokenizer

        dataset = self.test_dataset

        if data_path:
            self.test_data_path = data_path
            self._init_dataset()

        if dataset:
            dataset = dataset

        infer_results = []

        model.eval()
        with torch.inference_mode():
            for data in tqdm(dataset):
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
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()

                infer_results.append({"id": _id, "hint": generated_text})

        output = pd.DataFrame(infer_results)
        #output.hint = output.hint.map(lambda x: 0 if x.strip() == "" else x)
        #output.to_csv(self.data_path + "_hint.csv", index=False)
        output.to_csv(self.test_data_path[:-4] + "_hint.csv", index=False)

    def print_generate(self, dataset=None, n=5):
        model = self.model
        tokenizer = self.tokenizer

        dataset = self.test_dataset
        
        if dataset:
            dataset = dataset
        infer_results = []

        c=0
        model.eval()
        with torch.inference_mode():
            for data in tqdm(dataset):
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
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()

                infer_results.append({"id": _id, "answer": generated_text})

                c+=1

                if c==n:break

        print(infer_results)        

    def save(self):
        self.trainer.save_model(self.output_dir+self.model_name)
        self.tokenizer.save_pretrained(self.output_dir+self.model_name)

if __name__ == "__main__":
    config = None
    with open("../config/model.yaml") as f:
        config = yaml.safe_load(f)
    
    llm = LLM(config)
    llm.test()
        