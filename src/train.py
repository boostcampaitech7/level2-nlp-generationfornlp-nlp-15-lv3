import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from ast import literal_eval
import pandas as pd
import numpy as np
import random
import evaluate
from constants import *

# 난수 고정
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Config 로드
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# 모델과 Tokenizer 로드
def load_model_and_tokenizer(config):
    model_name = config["model"]["name"]
    trust_remote_code = config["model"]["trust_remote_code"]

    # 모델과 토크나이저 초기화 (FP16 설정)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,  # FP16 활성화
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Chat Template 설정
    tokenizer.chat_template = (
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
        "{% if system_message is defined %}{{ system_message }}{% endif %}"
        "{% for message in messages %}{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
    )

    # PEFT 설정 확인 및 적용
    if config.get("peft", {}).get("enable", True):
        lora_config = LoraConfig(
            r=config["peft"]["r"],
            lora_alpha=config["peft"]["lora_alpha"],
            lora_dropout=config["peft"]["lora_dropout"],
            target_modules=config["peft"]["target_modules"],
            task_type=config["peft"]["task_type"],
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer

# 데이터 전처리
def preprocess_dataset(config):
    dataset_path = config["data"]["dataset_path"]
    dataset = pd.read_csv(dataset_path)

    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": str(problems.get("answer", "")),  # 정답을 문자열로 저장
            "question_plus": problems.get("question_plus", None),
        }
        records.append(record)

    df = pd.DataFrame(records)
    df["question_plus"] = df["question_plus"].fillna("")

    PROMPT_NO_QUESTION_PLUS = """지문:
    {paragraph}

    질문:
    {question}

    선택지:
    {choices}

    1, 2, 3, 4, 5 중에서 하나를 골라 텍스트로 답하시오.
    정답:"""

    PROMPT_QUESTION_PLUS = """지문:
    {paragraph}

    질문:
    {question}

    <보기>:
    {question_plus}

    선택지:
    {choices}

    1, 2, 3, 4, 5 중에서 하나를 골라 텍스트로 답하시오.
    정답:"""

    processed_dataset = []
    for _, row in df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        processed_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": str(row["answer"])},
                ],
                "label": row["answer"],
            }
        )
    return Dataset.from_list(processed_dataset)

# 데이터 토큰화
def tokenize_dataset(processed_dataset, tokenizer, config):
    max_seq_length = config["data"]["max_seq_length"]

    def tokenize_function(example):
        formatted_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        outputs = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        return outputs

    tokenized_dataset = processed_dataset.map(
        tokenize_function, batched=True, remove_columns=["messages", "label"], desc="Tokenizing"
    )
    return tokenized_dataset.train_test_split(test_size=config["data"]["test_split_ratio"], seed=config["seed"])

# Trainer 설정
def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="<start_of_turn>model",
        tokenizer=tokenizer,
    )

    # 평가 함수 정의
    def preprocess_logits_for_metrics(logits, labels):
        choice_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 6)]
        logits = logits[:, -1, choice_token_ids]  # 마지막 토큰 기준
        return logits


    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # 레이블 데이터 처리
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # -100은 무시하도록 설정
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  # 디코딩
        # "<end_of_turn>" 같은 불필요한 텍스트 제거
        labels = [label.split("<end_of_turn>")[0].strip() for label in labels]
        # 정수로 변환하여 references 생성
        references = [int(label) - 1 for label in labels]

        # logits 데이터 처리
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)  # 소프트맥스 적용
        predictions = np.argmax(probs, axis=-1)  # 예측값 추출

        # 정확도 계산
        acc_metric = evaluate.load("accuracy")
        return acc_metric.compute(predictions=predictions, references=references)


    trainer_config = SFTConfig(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"]["train"],
        per_device_eval_batch_size=config["training"]["batch_size"]["eval"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy=config["training"]["eval_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["training"]["fp16"],
    )

    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=trainer_config,
    )

if __name__ == "__main__":
    config = load_config(TRAIN_CONFIG)
    set_seed(config["seed"])

    model, tokenizer = load_model_and_tokenizer(config)

    processed_dataset = preprocess_dataset(config)
    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer, config)

    train_dataset, eval_dataset = tokenized_dataset["train"], tokenized_dataset["test"]

    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    trainer.train()
