from tqdm import tqdm
import yaml
import torch
from peft import get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import numpy as np

import pandas as pd

from constants import *
import utils
from dataprocessor import DataProcessor

utils.set_seed(42)
class inferrer():
    def __init__(self,eval_args:dict):        
        #모델, 토크나이저 로드
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            eval_args['model'],
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            eval_args['model'],
            trust_remote_code=True,
        )
        
        
        dataloader = DataProcessor(datapath=eval_args['test_data'], tokenizer=self.tokenizer)


        infer_results = []

        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

        self.model.eval()
        with torch.inference_mode():
            for data in tqdm(dataloader.dataset):
                    _id = data["id"]
                    messages = data["messages"]
                    len_choices = data["len_choices"]

                    outputs = self.model(
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to("cuda")
                    )
                    logits = outputs.logits[:, -1].flatten().cpu()

                    target_logit_list = [logits[self.tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

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
                
        self.save_output(infer_results,eval_args['outputs_path'])
        
    def save_output(self, output:list, path:str = None):
        if path == None:
            path = BASE_DIR
        pd.DataFrame(output).to_csv(path+"/new_output.csv", index=False)
    
