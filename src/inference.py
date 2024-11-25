import torch
import pandas as pd
import numpy as np
from ast import literal_eval
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml
from constants import *


def load_config(config_path):
    # YAML 파일에서 설정을 로드
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(config):
    # 체크포인트에서 학습된 모델과 토크나이저를 로드
    checkpoint_path = config["checkpoint"]["path"]
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    return model, tokenizer


def preprocess_test_data(config):
    # 테스트 데이터셋을 로드하고 전처리
    test_data_path = config["data"]["test_path"]
    test_df = pd.read_csv(test_data_path)

    # JSON 구조를 평탄화
    records = []
    for _, row in test_df.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }
        records.append(record)

    test_df = pd.DataFrame(records)
    return test_df


def create_test_dataset(test_df, prompt_question_plus, prompt_no_question_plus):
    # 테스트 데이터
    test_dataset = []
    for _, row in test_df.iterrows():
        # 선택지 문자열 생성
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])

        # <보기>가 있을 때와 없을 때에 따른 메시지 생성
        if row["question_plus"]:
            user_message = prompt_question_plus.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = prompt_no_question_plus.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )
    return test_dataset


def inference(model, tokenizer, test_dataset, output_path):
    # 테스트 데이터에 대해 추론을 수행하고 결과를 저장
    infer_results = []

    model.eval()  # 모델을 평가 모드로 설정
    with torch.inference_mode():  # 추론 모드 활성화 (Gradient 계산 비활성화)
        for data in tqdm(test_dataset):  # 진행 상황을 표시하며 데이터 순회
            _id = data["id"]
            messages = data["messages"]

            # 입력 데이터를 토크나이저로 변환
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # Tensor로 처리
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs.to("cuda")  # GPU로 이동
                attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long).to("cuda")
            elif isinstance(inputs, dict) and "input_ids" in inputs:
                input_ids = inputs["input_ids"].to("cuda")
                attention_mask = inputs.get(
                    "attention_mask",
                    (input_ids != tokenizer.pad_token_id).to(torch.long).to("cuda"),
                )
            else:
                print("Error: Unexpected inputs format")
                print(inputs)  # 디버깅용 전체 출력
                continue

            # 모델로 텍스트 생성
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 명시적으로 attention_mask 전달
                max_new_tokens=50,  # 생성할 최대 토큰 수 (필요에 따라 조정)
                temperature=0.7,  # 텍스트 다양성을 조정
                top_p=0.9,  # nucleus sampling 비율
                do_sample=True,  # 샘플링 활성화
                num_return_sequences=1,  # 한 번에 생성할 출력 수
            )

            # 생성된 텍스트를 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 예측값 추출 (생성된 텍스트에서 특정 포맷을 찾는 방식 사용)
            predict_value = None
            for choice in range(1, data["len_choices"] + 1):
                if str(choice) in generated_text:
                    predict_value = str(choice)
                    break

            # 예측값을 저장
            infer_results.append({"id": _id, "answer": predict_value})

    # 결과를 CSV 파일로 저장
    pd.DataFrame(infer_results).to_csv(output_path, index=False)
    return infer_results




if __name__ == "__main__":
    # 설정 로드
    config = load_config(INFERENCE_CONFIG)

    # 모델과 토크나이저 로드
    model, tokenizer = load_model(config)

    # 테스트 데이터 로드 및 전처리
    test_df = preprocess_test_data(config)

    # 테스트 데이터셋 생성
    PROMPT_QUESTION_PLUS = config["prompts"]["prompt_question_plus"]
    PROMPT_NO_QUESTION_PLUS = config["prompts"]["prompt_no_question_plus"]
    test_dataset = create_test_dataset(test_df, PROMPT_QUESTION_PLUS, PROMPT_NO_QUESTION_PLUS)

    # 추론 수행
    output_path = config["output"]["path"]
    inference_results = inference(model, tokenizer, test_dataset, output_path)

    # 결과 출력
    print(pd.DataFrame(inference_results))
