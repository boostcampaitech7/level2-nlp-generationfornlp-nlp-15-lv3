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

test1 = pd.read_csv("../data/output_qwen_base_1126.csv")
test2 = pd.read_csv("../data/output_qwen_base_1127.csv")
test3 = pd.read_csv("../data/output_qwen_hint_1126.csv")
test4 = pd.read_csv("../data/output_qwen_hint_1127.csv")
test5 = pd.read_csv("../data/output_qwen_base_expected.csv")
test6 = pd.read_csv("../data/output_qwen_hint_expected.csv")
test7 = pd.read_csv("../data/output_qwen_base_1127_2.csv")

print("yesterday vs today qwen")
compare_answer(test1, test2)
print()

print("today vs today qwen")
compare_answer(test2, test7)
print()


print("yesterday vs today qwen with hint")
compare_answer(test3, test4)
print()


print("yesterday qwen vs qwen with hint")
compare_answer(test1, test3)
print()

print("today qwen vs qwen with hint")
compare_answer(test2, test4)
print()

"""
print("submitted vs today")
compare_answer(test2, test5)
compare_answer(test4, test6)
print()
compare_answer(test4, test7)
"""