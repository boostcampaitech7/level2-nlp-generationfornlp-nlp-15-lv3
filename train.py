import yaml
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import BitsAndBytesConfig
from ast import literal_eval
import evaluate

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def load_model(config):
    model_name = config['model']['name']
    trust_remote_code = config['model']['trust_remote_code']

    # 양자화 설정
    quantization_config = None
    if config.get('quantization', {}).get('enable', False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=config['quantization']['fp32_cpu_offload']
        )

    # Load base model and tokenizer with optional quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # Apply PEFT if enabled
    if config['peft']['enable']:
        peft_config = LoraConfig(
            r=config['peft']['r'],
            lora_alpha=config['peft']['lora_alpha'],
            lora_dropout=config['peft']['lora_dropout'],
            target_modules=config['peft']['target_modules'],
            task_type=config['peft']['task_type']
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer

def preprocess_dataset(config, tokenizer):
    dataset_path = config['data']['dataset_path']
    max_seq_length = config['data']['max_seq_length']
    test_split_ratio = config['data']['test_split_ratio']

    dataset = pd.read_csv(dataset_path)
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            'question_plus': problems.get('question_plus', None),
        }
        records.append(record)

    df = pd.DataFrame(records)
    df['question_plus'] = df['question_plus'].fillna('')
    df['full_question'] = df.apply(
        lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'], axis=1
    )
    processed_dataset = Dataset.from_pandas(df)

    def tokenize(example):
        outputs = tokenizer(
            example['full_question'],
            truncation=True,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        return outputs

    tokenized_dataset = processed_dataset.map(
        tokenize,
        batched=True,
        desc="Tokenizing"
    )
    return tokenized_dataset.train_test_split(test_size=test_split_ratio, seed=config['seed'])

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    output_dir = config['training']['output_dir']
    batch_size_train = config['training']['batch_size']['train']
    batch_size_eval = config['training']['batch_size']['eval']

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template="<start_of_turn>model",
        tokenizer=tokenizer
    )

    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions = np.argmax(logits, axis=-1)
        return acc_metric.compute(predictions=predictions, references=labels)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        num_train_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        eval_strategy=config['training']['eval_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=sft_config,
    )
    return trainer

if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config['seed'])

    # Load model and tokenizer
    model, tokenizer = load_model(config)

    # Preprocess dataset
    datasets = preprocess_dataset(config, tokenizer)
    train_dataset, eval_dataset = datasets['train'], datasets['test']

    # Setup trainer and train
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    trainer.train()

