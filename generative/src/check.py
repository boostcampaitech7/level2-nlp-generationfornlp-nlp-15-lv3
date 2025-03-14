from utils import *
import pandas as pd
import os 

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
'''
test1 = pd.read_csv("../data/output_qwen_base_1126.csv")
test2 = pd.read_csv("../data/output_qwen_base_1127.csv")
test3 = pd.read_csv("../data/output_qwen_hint_1126.csv")
test4 = pd.read_csv("../data/output_qwen_hint_1127.csv")
test5 = pd.read_csv("../data/output_qwen_base_expected.csv")
test6 = pd.read_csv("../data/output_qwen_hint_expected.csv")
test7 = pd.read_csv("../data/output_qwen_base_1127_2.csv")
test8 = pd.read_csv("../data/output.csv")
'''
"""
test1 = pd.read_csv("../data/output_rag.csv")
test2 = pd.read_csv("../data/output_rag2.csv")

print("")
compare_answer(test1, test2)
print()
"""
"""
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
"""
print("submitted vs today")
compare_answer(test2, test5)
compare_answer(test4, test6)
print()
compare_answer(test4, test7)
"""

"""
cls = pd.read_csv("../data/output_cls.csv")
test = pd.read_csv("../data/test.csv")
merge(test, cls).to_csv("../data/test.csv", index=False)
"""

"""
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
"""
"""
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
            if id_ in rag.id.values:
                results += [{"id": id_, "answer": rag[rag.id == id_].values[0][1]}]
                c+=1
            elif id_ in pred.id.values:
                results += [{"id": id_, "answer": pred[pred.id == id_].values[0][1]}]
                c+=1

        print(c)

        output = pd.DataFrame(results)
        print(output)
        output.to_csv("../data/output_final.csv", index=False)
"""
test = pd.read_csv("../data/output_test.csv")
test1 = pd.read_csv("../data/output_final.csv")
compare_answer(test, test1)