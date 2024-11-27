import pandas as pd
import re

from langchain.prompts import PromptTemplate
from googletrans import Translator

korean = r"ㄱ-ㅎㅏ-ㅣ가-힣"
english = r"a-zA-Z"
number = r"0-9"
space = r"\s"
punctuation = r".,"
asciicode = r"\x00-\x7F"
flags = re.UNICODE

def remove_not_numeric(text):
    text = re.sub(f"[^{number}]", "1", text, flags)
    return text

def remove_not_digital(text):
    text = re.sub(r'\b\d{2,}\b', '1', text)
    return text

def remain_numeric(df_a):
    df_a.answer = df_a.answer.map(remove_not_numeric)

def calculate_ratio_korean(text):
    length_text = len(text)
    korean_text = re.sub(f"[^{korean}]", "", text)
    length_korean_text = len(korean_text)
    return length_korean_text / length_text


def translate(text, src, dest):
    translator = Translator()
    return translator.translate(text, src=src, dest=dest).text

def compare_answer(df_a, df_b):
    c = 0
    for a, b in zip(df_a.answer, df_b.answer):
        if a==b: c+=1
    print("each len:", len(df_a.answer), len(df_b.answer))
    print("& len:", c)

def compare_answer_old(df_a, df_b):
    c = 0
    for a, b in zip(df_a.answer, df_b.answer):
        if str(b) in a: c+=1
    print("each len:", len(df_a.answer), len(df_b.answer))
    print("& len:", c)

def check_answer(df):
    c = 0
    for a in df.answer:
        if not isinstance(a, int):
            c+=1
    print("num of not int:", c)

def merge(df_a, df_b):
    df = df_a.merge(df_b)
    return df


def eval():
    data_path = '../data/'
    train_data_path = data_path + "train.csv"
    test_data_path = data_path + "test.csv"

    dataset = pd.read_csv(train_data_path) 

    #dataset.train_test_split(test_size=0.1, seed=42)

    print(dataset)

def get_prediction_prompt():
    PROMPT_NO_QUESTION_PLUS = """
        지문:
        {paragraph}

        질문:
        {question}

        선택지:
        {choices}

        힌트:
        {hint}

        지문을 읽고 모든 지식과 논리적 추론을 통해 문제를 해결해라.
        지문과 힌트를 이용하여 질문에 대한 가장 적합한 정답을 선택지에서 찾아라.  
        정답은 논리적 추론을 통해 결정되며, 힌트가 제공된 경우 이를 활용하여 더 정확하게 정답을 골라라.

        위 선택지 1, 2, 3, 4, 5 중에서 가장 적합한 정답 번호를 하나 골라라. 
        반드시 논리적 근거를 바탕으로 정답 번호를 골라라.
        
        설명과 분석은 생략하고 정답 번호부터 바로 말하자.

        정답:
        """
    
    PROMPT_QUESTION_PLUS = """
        지문:
        {paragraph}

        질문:
        {question}

        <보기>:
        {question_plus}

        선택지:
        {choices}

        힌트:
        {hint}

        지문을 읽고 모든 지식과 논리적 추론을 통해 문제를 해결해라.  
        지문과 힌트를 이용하여 질문에 대한 가장 적합한 정답을 선택지에서 찾아라.  
        정답은 논리적 추론을 통해 결정되며, 힌트가 제공된 경우 이를 활용하여 더 정확하게 정답을 골라라.

        위 선택지 1, 2, 3, 4, 5 중에서 가장 적합한 정답 번호를 하나 골라라. 
        반드시 논리적 근거를 바탕으로 정답 번호를 골라라.
        
        설명과 분석은 생략하고 정답 번호부터 바로 말하자.

        정답:
        """
    
    PROMPT_SYSTEM = "당신은 전지전능한 신으로 모든 걸 알고 있다. 당신은 객관식 문제를 풀게 될 것이고 말할 때도 오직 정답 번호 하나만 말한다. 수능 지문과 질문과 선택지와 힌트가 주어진다. 정답의 근거는 설명하지 마라. 오직 정답만 말해라. 질문의 정확한 정답인 숫자 하나만 반드시 말하라."

    return PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, PROMPT_SYSTEM


def get_hint_prompt():
    PROMPT_NO_QUESTION_PLUS = """
        지문을 읽고 학생들이 질문에 대한 답을 스스로 유추할 수 있도록 돕는 힌트를 제공합니다.
        힌트는 지문에서 중요한 부분을 강조하거나, 답을 찾는 논리적 사고 과정을 안내하는 방식으로 작성합니다.
        힌트는 100자 이내로 작성합니다.
    
        지문:
        {paragraph}

        질문:
        {question}

        선택지:
        {choices}

        힌트:"""

    PROMPT_QUESTION_PLUS = """
        지문을 읽고 학생들이 질문에 대한 답을 스스로 유추할 수 있도록 돕는 힌트를 제공합니다.
        힌트는 지문에서 중요한 부분을 강조하거나, 답을 찾는 논리적 사고 과정을 안내하는 방식으로 작성합니다.
        힌트는 100자 이내로 작성합니다.
        
        지문:
        {paragraph}

        질문:
        {question}

        <보기>:
        {question_plus}

        선택지:
        {choices}

        힌트:"""
    
    PROMPT_SYSTEM = "당신은 전지전능한 신으로 모든 걸 알고 있다. 수능 지문과 질문과 정답이 주어졌을 때, 또는 정답이 없더라도 어떤 정답이 나올 수 있도록 힌트를 작성하라. 바로 본론부터 시작하라."

    return PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, PROMPT_SYSTEM

def get_rag_prompt():
    prompt = PromptTemplate(
                input_variables = ["context", "input"],
                template = """
                    다음 정보를 참고하며 질문에 올바른 정답을 말해라.
                    정보가 질문과 상관없으면 참고하지 않아도 된다.
                    
                    정보 : {context}
                    
                    질문 : {input}
                """
            )

    return prompt