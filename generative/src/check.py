from utils import *
import pandas as pd

"""
gen = pd.read_csv("../data/output_gen.csv")
logits = pd.read_csv("../data/output_log.csv")
test = pd.read_csv("../data/output.csv")
gen_prompt = pd.read_csv("../data/output_gen_prompt.csv")
logits_prompt = pd.read_csv("../data/output_log_prompt.csv")
train = pd.read_csv("../data/train.csv")
hint = pd.read_csv("../data/hint.csv")
"""
llama_output = pd.read_csv("../data/output.csv")
llama_output.answer = llama_output.answer.map(remove_not_numeric).map(remove_not_digital).astype(pd.Int32Dtype())
print(llama_output.answer.dtype)
llama_output.to_csv("../data/output_llama_1b_hint.csv", index=False)

#train_hint = merge(train, hint)
#train_hint.to_csv("../data/train_hint.csv", index=False)

#compare_answer(logits_prompt, test)
#check_answer(test)