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

from utils import *
from trainer import *

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

if __name__ == "__main__":
    config = None
    with open("../config/model.yaml") as f:
        config = yaml.safe_load(f)
    

    print(config)

    if config['task'] == "question_classification":
        print(config)
        PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, SYSTEM_PROMPT = get_classification_prompt()

        dp = DataProcessor(config)
        dp.init()
        dp.set_system_prompt(SYSTEM_PROMPT)
        dp.set_user_prompt(PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS)

        train_dataset = dp.get_train_dataset()
        test_dataset = dp.get_test_dataset()
            
        model = get_model(config)
        tokenizer = get_tokenizer(config)
        collator = get_collator(config, tokenizer)

        if config['use_gradient_checkpointing']:
            model.gradient_checkpointing_enable()

        llm = LLM(config)
        llm.set_model(model)
        llm.set_tokenizer(tokenizer)
        llm.set_collator(collator)
        llm.set_train_dataset(tokenize_dataset(tokenizer, train_dataset))
        llm.set_test_dataset(test_dataset)
        llm.init()

        llm.test()

        
        cls = pd.read_csv("../data/output_cls.csv")
        test = pd.read_csv("../data/test.csv")
        if "class" not in test.columns:
            print("add class to test")
            merge(test, cls).to_csv("../data/test.csv", index=False)
            print("classification done")


    if config['task'] == "hint_generation":
        PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, SYSTEM_PROMPT = get_hint_prompt()

        dp = DataProcessor(config)
        dp.init()
        dp.set_system_prompt(SYSTEM_PROMPT)
        dp.set_user_prompt(PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS)

        train_dataset = dp.get_train_dataset()
        test_dataset = dp.get_test_dataset()
        
        model = get_model(config)
        tokenizer = get_tokenizer(config)
        collator = get_collator(config, tokenizer)

        if config['use_gradient_checkpointing']:
            model.gradient_checkpointing_enable()

        llm = LLM(config)
        llm.set_model(model)
        llm.set_tokenizer(tokenizer)
        llm.set_collator(collator)
        llm.set_train_dataset(tokenize_dataset(tokenizer, train_dataset))
        llm.set_test_dataset(test_dataset)
        llm.init()

        llm.print_generate()
        llm.generate()

        torch.cuda.empty_cache()
        print("hint generation done")

    elif config['task'] == "prediction_generation":
        if config['use_hint']:
            train = pd.read_csv("../data/train.csv")
            test = pd.read_csv("../data/test.csv")

            if os.path.isfile("../data/train_hint.csv"):
                train_hint = pd.read_csv("../data/train_hint.csv")
                train_hint.hint = train_hint.hint.map(lambda hint:translate(hint, 'auto', 'ko') if 0.0 < calculate_ratio_korean(hint) < 0.3 else hint)
                merge(train, train_hint).to_csv("../data/train_merge_hint.csv", index=False)
                config['train_data_file'] = "train_merge_hint.csv"

            if os.path.isfile("../data/test_hint.csv"):
                test_hint = pd.read_csv("../data/test_hint.csv")
                test_hint.hint = test_hint.hint.map(lambda hint:translate(hint, 'auto', 'ko') if 0.0 < calculate_ratio_korean(hint) < 0.3 else hint)
                merge(test, test_hint).to_csv("../data/test_merge_hint.csv", index=False)
                config['test_data_file'] = "test_merge_hint.csv"

        
        print(config)
        PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, SYSTEM_PROMPT = get_prediction_prompt()

        dp = DataProcessor(config)
        dp.init()
        dp.set_system_prompt(SYSTEM_PROMPT)
        dp.set_user_prompt(PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS)

        train_dataset = dp.get_train_dataset()
        test_dataset = dp.get_test_dataset()
            
            
        if not config['test_rag']:
            model = get_model(config)
            tokenizer = get_tokenizer(config)
            collator = get_collator(config, tokenizer)

            if config['use_gradient_checkpointing']:
                model.gradient_checkpointing_enable()

            llm = LLM(config)
            llm.set_model(model)
            llm.set_tokenizer(tokenizer)
            llm.set_collator(collator)
            llm.set_train_dataset(tokenize_dataset(tokenizer, train_dataset))
            llm.set_test_dataset(test_dataset)
            llm.init()

            if config['train']:
                llm.train()
            if config['eval']:
                llm.evaluate()
            if config['test']: 
                llm.test()
                print("prediction done")

            del model
            del llm

            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
        if config['test_rag']:
            from rag import RAG
            with open('../config/rag.yaml', 'r') as f:
                config = yaml.safe_load(f)
            rag = RAG(config)
            rag.set_prompt(get_rag_prompt())
            rag.set_combine_docs_chain()
            rag.set_rag_chain()


            tokenizer = get_tokenizer(config)

            infer_results = []

            with torch.inference_mode():
                for data in tqdm(test_dataset):
                    if "class" in data and data["class"] == 1:print("rag) class 1 skip");continue
                    _id = data["id"]
                    messages = data["messages"]
                    len_choices = data["len_choices"]

                    input = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to("cuda")
                    
                    inputs = tokenizer.batch_decode(input)[0].strip()
                    outputs = rag.invoke(inputs)["answer"][1]

                    infer_results.append({"id": _id, "answer": remove_not_digital(remove_not_numeric(outputs))})

            output = pd.DataFrame(infer_results)
            print(output)
            output.answer = output.answer.map(lambda x: 0 if x.strip() == "" else x)
            output.to_csv(config['data_path'] + "/output_rag.csv", index=False)

            print("rag prediction done")
    
    if os.path.isfile("../data/test.csv") and os.path.isfile("../data/output_test.csv") and os.path.isfile("../data/output_rag.csv"):
        test = pd.read_csv("../data/test.csv")
        pred = pd.read_csv("../data/output_test.csv")
        rag = pd.read_csv("../data/output_rag.csv")

        if (len(set(test.id)) != len(set(pred.id) | set(rag.id))) and (len(set(test.id)) == len(set(pred.id)) + len(set(rag.id))):
            print("test, pred, rag data size not matched")
        else:
            c=0
            results = []
            for id_ in test.id:
                if id_ in pred.id.values:
                    results += [{"id": id_, "answer": pred[pred.id == id_].values[0][1]}]
                    c+=1
                elif id_ in rag.id.values:
                    results += [{"id": id_, "answer": rag[rag.id == id_].values[0][1]}]
                    c+=1
            print(c)

            output = pd.DataFrame(results)
            print(output)
            output.to_csv("../data/output_final.csv", index=False)
