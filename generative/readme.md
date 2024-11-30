# 수능형 문제 풀이 모델 생성
## Performance
### Metrics 
Accuracy

### baseline

<img src='./images/baseline.png' height='128'>

### Qwen-32B(int4) + test_hint(Qwen-32B)

<img src='./images/hint.png' height='128'>

### Qwen-32B(int4) + test_hint(Qwen-32B) + RAG

<img src='./images/rag.png' height='128'>

## Usage
### Requirements
```bash
pip install -r requirements.txt
```

### Command
```bash
cd ./src
python run.py
```