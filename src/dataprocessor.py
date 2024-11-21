import pandas as pd
from ast import literal_eval
from datasets import Dataset

from constants import PROMPT_QUESTION_PLUS, PROMPT_NO_QUESTION_PLUS


class DataProcessor():
    def __init__(self, datapath:str, tokenizer):
        self.tokenizer = tokenizer
        
        dataset = pd.read_csv(datapath) 
        
        #TODO: 변수명 정리, 파이프라인 별도 함수화 필요
        # Flatten the JSON dataset
        records = []
        for _, row in dataset.iterrows():
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
        df = pd.DataFrame(records)
        
        # Combine 'question' and 'question_plus' if available
        df['question_plus'] = df['question_plus'].fillna('')
        df['full_question'] = df.apply(lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'], axis=1)

        # Calculate the length of each question
        df['question_length'] = df['full_question'].apply(len)
        
                       
        processed_dataset = []
        for _, row in df.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            len_choices = len(row["choices"])
            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                )

            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{row['answer']}"}
                    ],
                    "label": row["answer"],
                    "len_choices": len_choices, 
                }
            )
        processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
        
        print(f"Type of processed_dataset before conversion: {type(processed_dataset)}")

        self.dataset = processed_dataset
        self.processed_dataset = self.tokenize(processed_dataset)
        
        return
    
    def formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def tokenize_func(self, element):
        outputs = self.tokenizer(
            self.formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def tokenize(self, dataset):
        #TODO: tokenize랑 tokenize_func 이름을 바꾸던 합치던 해야
        # 데이터 토큰화
        dataset = dataset.map(
            self.tokenize_func,
            remove_columns=list(dataset.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
        return dataset
    
    def data_separator(self, test_size):
        dataset = self.processed_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)  
        dataset = self.processed_dataset.train_test_split(test_size=test_size, seed=42)

        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        return train_dataset, eval_dataset
