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
from trainer import LLM

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
    
    if config['use_hint']:
        train = pd.read_csv("../data/train.csv")
        train_hint = pd.read_csv("../data/train_hint.csv")
        test = pd.read_csv("../data/test.csv")
        test_hint = pd.read_csv("../data/test_hint.csv")

        #train_merge_hint = merge(train, train_hint)
        #train_merge_hint.to_csv("../data/train_merge_hint.csv", index=False)
        merge(train, train_hint).to_csv("../data/train_merge_hint.csv", index=False)
        merge(test, test_hint).to_csv("../data/test_merge_hint.csv", index=False)

        config['train_data_file'] = "train_merge_hint.csv"
        config['test_data_file'] = "test_merge_hint.csv"

    from trainer import *


    
    print(config)

    PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, SYSTEM_PROMPT = get_prediction_prompt() if config['task'] == "prediction_generation" else get_hint_prompt()

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
    if config['train']:
        llm.train()
    if config['eval']:
        llm.evaluate()
    if config['test']: 
        llm.test()
    if config['generate']:
        llm.print_generate()
        llm.generate()