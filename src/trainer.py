import yaml
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.model = AutoModelForCausalLM.from_pretrained(
        train_args['model'],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        
        )
        self.set_tokenizer(train_args['tokenizer'])

        #데이터셋 로드
        dataprocessor = DataProcessor(train_args['train_data'],self.tokenizer)
        train_dataset, eval_dataset = dataprocessor.data_separator(0.1)
        
        #LoRA 설정
        peft_config = LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.set_metrics()
        
        #트레이너 설정
        sft_config = SFTConfig(
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine",
            max_seq_length=1024,
            output_dir=train_args['outputs_path'], 
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            save_only_model=True,
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator("<start_of_turn>model"),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=sft_config,
        )
        
        return
    
    def train(self):
        self.trainer.train()
        return
    
    def set_tokenizer(self, tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            trust_remote_code=True,
        )
        tokenizer.chat_template = CHAT_TEMPLATE
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'right'
        self.tokenizer = tokenizer        
        return
    
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
        
