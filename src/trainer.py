import yaml
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import numpy as np
import evaluate
import utils

import os

from constants import *
from dataprocessor import DataProcessor

utils.set_seed(42)

class MainTrainer():
    def __init__(self, train_args:dict):
        #모델, 토크나이저 로드
        self.model =     self.get_model(train_args.get("model"))
        self.tokenizer = self.get_tokenizer(train_args.get("model").get('tokenizer'))

        #데이터셋 로드
        dataprocessor = DataProcessor(train_args.get("data"),self.tokenizer)
        train_dataset, eval_dataset = dataprocessor.get_separated()
        
      
        self.set_metrics()
        
        #트레이너 설정
        sft_config = self.get_sft_config(train_args.get("training"))
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator("<start_of_turn>model"),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            args=sft_config,
        )
        
        return
      
    
    def train(self):
        self.trainer.train()
        return
    
    def get_tokenizer(self, config):
        tokenizer = AutoTokenizer.from_pretrained(
            config['name'],
            trust_remote_code=True,
        )
        tokenizer.chat_template = config['chat_template']
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = config['padding_side']        
        return tokenizer
    
    def get_model(self, config):
        model_name = config["name"]
        trust_remote_code = config["trust_remote_code"]

        # YAML에서 양자화 설정 가져오기
        quantization = config.get("quantization", {})
        if quantization.get("enable", False):  # 양자화가 활성화된 경우
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantization.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=torch.__dict__.get(quantization.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_use_double_quant=quantization.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config,
                config={"hidden_activation": "gelu_pytorch_tanh"},  # hidden_activation 명시
            )
        else:  # 양자화가 비활성화된 경우
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                config={"hidden_activation": "gelu_pytorch_tanh"},  # hidden_activation 명시
            )
            
        # PEFT 설정
        peft = config.get("peft", {})
        if peft.get("enable", True):
            lora_config = LoraConfig(
                r=peft["r"],
                lora_alpha=peft["lora_alpha"],
                lora_dropout=peft["lora_dropout"],
                target_modules=peft["target_modules"],
                task_type=peft["task_type"],
                bias=peft["bias"]
            )
            model = get_peft_model(model, lora_config)
        
        return model

    
    def set_metrics(self):
        #TODO metric 설정 좀 더 확실하게
        self.acc_metric = evaluate.load("accuracy")
        
        # 정답 토큰 매핑
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        
        return
    
    def compute_metrics(self, evaluation_result):
        logits, labels = evaluation_result

        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: self.int_output_map[x], labels))

        # 소프트맥스 함수를 사용하여 로그트 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits).float(), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 정확도 계산
        acc = self.acc_metric.compute(predictions=predictions, references=labels)
        return acc
    
    # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def preprocess_logits_for_metrics(self, logits, labels):
        tokenizer = self.tokenizer
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"], tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx] # -2: answer token, -1: eos token
        return logits
    
    def data_collator(self, response_template):
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )
    
    def get_sft_config(self, config):
        return SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=config["batch_size"]["train"],
        per_device_eval_batch_size=config["batch_size"]["eval"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        eval_strategy=config["eval_strategy"],
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        )
        
