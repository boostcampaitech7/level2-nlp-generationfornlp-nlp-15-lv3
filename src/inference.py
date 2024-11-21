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
    #YAML 파일에서 설정을 로드
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(config):
    #체크포인트에서 학습된 모델과 토크나이저를 로드
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
    #테스트 데이터셋을 로드하고 전처리
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
    #테스트 데이터
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
    #테스트 데이터에 대해 추론을 수행하고 결과를 저장.
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}  # 예측 값 매핑

    model.eval()  # 모델을 평가 모드로 설정
    with torch.inference_mode():  # 추론 모드 활성화 (Gradient 계산 비활성화)
        for data in tqdm(test_dataset):  # 진행 상황을 표시하며 데이터 순회
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            # 입력 데이터를 토크나이저로 변환
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")  # GPU로 이동

            # 모델 추론
            outputs = model(inputs)
            logits = outputs.logits[:, -1].flatten().cpu()  # 마지막 출력 로짓 가져오기

            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

            # 소프트맥스를 통해 확률 계산
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(target_logit_list, dtype=torch.float32)
                )
                .detach()
                .cpu()
                .numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
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
