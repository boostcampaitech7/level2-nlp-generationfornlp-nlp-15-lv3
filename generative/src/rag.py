# %%
import torch
from torch.cuda.amp import autocast, GradScaler


import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset, load_dataset
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig

import os, time
import yaml

from huggingface_hub import login

import faiss

from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
import sentence_transformers

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



class RAG:
    def __init__(self, config=None):
        self.documents = []
        self.emb_name = None
        self.emb_model = None
        self.faiss_folder_path = "../data/faiss_index"
        self.faiss_store = None

        self.llm_name = None
        self.llm = None
        self.task = "text-generation"
        self.batch_size = 1
        self.device_map = "auto"
        self.max_new_tokens=200

        self.k = 1
        self.retrieval_qa_chat_prompt = None
        self.combine_docs_chain = None
        self.rag_chain = None

        self.set_config(config)
        self._init_documents()
        self._init_faiss_store()
        self._init_chain()

    def set_config(self, config=None):
        #self.documents = []
        self.emb_name = config['emb_name']
        self.faiss_folder_path = config['faiss_folder_path']

        self.llm_name = config['model_name'] #config['llm_name']
        self.task = config['task']
        self.batch_size = config['batch_size']
        self.device_map = config['device_map']
        self.max_new_tokens = config['max_new_tokens']

        self.k = config['k']

    def _init_documents(self):
        wiki = load_dataset("wiki40b", "ko")
        print(wiki)
        #print(eval(wiki['train']['text'][100]).decode('utf8'))

        documents = []
        # 추후에 train valid test 통합
        for text in wiki['train']['text']:
            for tag in ['_START_ARTICLE_', '_START_SECTION_', '_START_PARAGRAPH_', '_NEWLINE_']:
                text = text.replace(tag, ' ') 
            documents += [eval(text).decode('utf8')]

        start_time = time.perf_counter()
        self.documents = [Document(page_content=doc.strip()) for doc in documents]
        end_time = time.perf_counter()
        print("documents list:",end_time-start_time,"s")

    def _init_faiss_store(self):
        start_time = time.perf_counter()
        self.emb_model = HuggingFaceEmbeddings(model_name=self.emb_name)

        if not os.path.exists(self.faiss_folder_path):
            # FAISS index 생성
            print("creating faiss index")
            self.faiss_store = FAISS.from_documents(self.documents, self.emb_model)
            self.faiss_store.save_local(self.faiss_folder_path)
        else:
            # 저장된 FAISS 인덱스 불러오기
            print("loading faiss index")
            self.faiss_store = FAISS.load_local(self.faiss_folder_path, self.emb_model, allow_dangerous_deserialization=True)
        end_time = time.perf_counter()
        print("set faiss index:",end_time-start_time,"s")

    def _init_chain(self):
        self.set_llm()
        self.set_prompt()
        self.set_combine_docs_chain()
        self.set_rag_chain()
    
    def set_llm(self, llm_name=None, task=None, device_map=None, batch_size=None, max_new_tokens=None):
        if llm_name:
            self.llm_name=llm_name
        if task:
            self.task=task
        if device_map:
            self.device_map=device_map
        if batch_size:
            self.batch_size=batch_size
        if max_new_tokens:
            self.max_new_tokens=max_new_tokens

        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.llm_name,
            task=self.task,
            #device=0,  # replace with device_map="auto" to use the accelerate library.
            device_map=self.device_map,
            batch_size=self.batch_size,
            pipeline_kwargs={"max_new_tokens": self.max_new_tokens, "return_full_text" : False},
        )

    def set_prompt(self, prompt=None):
        if prompt:
            self.retrieval_qa_chat_prompt = prompt
        else:
            self.retrieval_qa_chat_prompt = PromptTemplate(
                input_variables = ["context", "input"],
                template = """
                    다음 정보를 참고하며 질문에 올바른 정답을 말해라.
                    
                    정보 : {context}
                    
                    질문 : {input}

                    정보가 질문과 상관없으면 참고하지 않아도 된다.
                    질문과 관련 없는 대답은 하지 마라.
                    정답을 반드시 말하라.
                    정답은 오직 하나만 말해라.

                    정답 : 
                """
            )

    def set_combine_docs_chain(self, llm=None, prompt=None):
        if llm:
            self.llm=llm
        if prompt:
            self.retrieval_qa_chat_prompt = prompt

        self.combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=self.retrieval_qa_chat_prompt
        )

    def set_rag_chain(self, docs_store=None, k=1, combine_docs_chain=None):
        if docs_store:
            self.faiss_store=docs_store
        if k!=1:
            self.k=1
        if combine_docs_chain:
            self.combine_docs_chain = combine_docs_chain

        self.rag_chain = create_retrieval_chain(
            retriever=self.faiss_store.as_retriever(search_kwargs={"k": self.k}),
            combine_docs_chain=self.combine_docs_chain
        )
            

    def invoke(self, query):
        return self.rag_chain.invoke({"input": query})
        
    def print_invoke(self, query):
        start_time = time.perf_counter()
        #query = ["test1", "test2"] #"이 질문은 다음 정보를 참조한다. 텡크테리족의 등장 이전에는 브루크테리족이 등장했지만, 현재 일반적으로는 차마비족과 앙그리바리족이 그들의 정착지에 들어가 쫓아낸 이후 [이웃] 부족의 도움으로 그들을 완전히 절멸시켰다고 여겨진다. 이들은 폭정에 대항하고자 행동했거나 약탈의 유혹 때문에, 또는 하늘의 [호의적인] 뜻에 따라 행동했던 것이다. 갈등의 광경으로 인한 원한은 일어나지도 않았다. 6만 명이 넘는 목숨이 로마의 병력과 무기 아래 쓰러진 것이 아니라 훨씬 더 거대한 것을 위해 위해 기꺼이 스러졌다. 부디 부족들이 우리에 대한 사랑은 아니더라도 서로에 대한 증오는 항상 간직하기를 기도한다. 운명은 제국의 손에 의해 우리가 몰락하게 만들었지만, 운명이 가져오는 적들 사이의 불화만큼 더 큰 축복은 존재하지 않기 때문이다.” 게르만족의 기원과 현황, 타키투스, 기원후 98년 경. 다음 중 로마 제국의 쇠퇴에 가장 적게 기여한 것은 무엇인가?', 'choices': ['왕좌를 노리고 벌어진 라이벌 간의 내전', '야만인 민족의 제국 침입', '제국의 정부 기관으로서 원로원의 지속성', '질병과 전염병으로 인한 인구학적 취약성']," #"소련이 한국에 진입하게 만든 당 시대의 세계사적 사건은?" #"이인은 선왕에게 제사를 드리고, 현왕을 할아버지의 사당 앞에 경건하게 바쳤다. . . . 이인은 이어서 어린 왕에게 교훈을 주기 위해 공덕조의 덕을 분명히 설명했다. 오! 옛날 하나라의 선왕들은 덕을 열심히 쌓았고, 그때는 하늘에서 재앙이 내리지 않았습니다. 산과 강의 정령들은 모두 고요했고, 새와 짐승들은 타고난 본성에 따라 존재 자체를 즐겼습니다. 그러나 후손들은 선대의 본보기를 따르지 않았고, 하늘은 총애하는 통치자를 이용하여 재앙을 내렸습니다. 하나라에 대한 공격은 명조의 난교에서 그 원인을 찾을 수 있습니다. 상나라의 왕은 현자와도 같은 능력을 눈부시게 펼쳤습니다. 압제가 아닌 너그러운 온화함으로 지배했기 때문입니다. 이제 폐하꼐서는 자신의 덕을 계승하고자 합니다. 모든 것은 통치를 어떻게 시작하느냐에 달렸습니다. “오! 선왕께서는 사람들을 하나로 묶는 유대감에 세심하게 주의를 기울이는 것으로 통치를 시작하셨나이다… 이러한 경고를 항상 염두에 두소서… 하늘의 길은 변함이 없습니다. 선행을 하면 축복을 내리고 악행을 하면 불행을 내립니다. 폐하께서 크든 작든 공덕을 쌓지 않으시면 조상들이 가꿔 온 왕조에 파멸을 불러 올 것입니다.” —기원전 6세기 중국 서경에서 발췌 및 각색 다음 중 도교 신앙의 뿌리라고 볼 수 있는 것은?'"
        result = self.rag_chain.invoke({"query": query})
        print("query:", query)
        print("answer:", result['answer'])
        end_time = time.perf_counter()
        print("rag chain run:", end_time-start_time,"s")
    
    def batch(self, queries):
        return self.rag_chain.batch(queries)['answer']
    
    def print_batch(self, queries):
        start_time = time.perf_counter()
        #result = self.rag_chain.batch([{"input": "도교 신앙의 뿌리라고 볼 수 있는 것은?"},{"input": "로마가 쇠퇴한 원인은?"}])
        result = self.rag_chain.batch(queries)
        print("queries:", queries)
        print("answer:", result)
        end_time = time.perf_counter()
        print("rag chain run:", end_time-start_time,"s")
    
if __name__ == "__main__":
    with open('../config/rag.yaml', 'r') as f:
        config = yaml.safe_load(f)
    rag = RAG(config)
    query = "한국은 언제 나왔지?"
    print("query:", query)
    print("answer:", rag.invoke(query))