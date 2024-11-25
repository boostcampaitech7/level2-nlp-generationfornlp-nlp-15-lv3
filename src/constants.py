import os

# 현재 파일의 절대 경로 (constants.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# constants.py 파일 바로 위의 디렉토리 경로
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
TRAIN_CONFIG = os.path.join(CONFIG_DIR, 'config_train.yaml')
INFERENCE_CONFIG = os.path.join(CONFIG_DIR, 'config_inference.yaml')


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