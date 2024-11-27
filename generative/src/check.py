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
"""
llama_output = pd.read_csv("../data/output.csv")
llama_output.answer = llama_output.answer.map(remove_not_numeric).map(remove_not_digital).astype(pd.Int32Dtype())
print(llama_output.answer.dtype)
llama_output.to_csv("../data/output_llama_1b_hint.csv", index=False)
"""
#train_hint = merge(train, hint)
#train_hint.to_csv("../data/train_hint.csv", index=False)

#compare_answer(logits_prompt, test)
#check_answer(test)
'''
test_hint = pd.read_csv("../data/test_hint.csv")
"""
for hint in test_hint.hint:
    trans_hint = translate(hint, 'auto', 'ko')
    if hint == trans_hint:continue
    print("koraen ratio:", calculate_ratio_korean(hint))
    print("origin:", hint)
    print("trans:", trans_hint)
    print()
    
    import time
    time.sleep(0.05)
"""
test_hint.hint = test_hint.hint.map(lambda hint:translate(hint, 'auto', 'ko') if 0.0 < calculate_ratio_korean(hint) < 0.3 else hint)
test_hint.to_csv("../data/test_hint_trans.csv", index=False)
'''

test1 = pd.read_csv("../data/output.csv")
test2 = pd.read_csv("../data/output_qwen_base.csv")
compare_answer(test1, test2)