import csv
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate import init_empty_weights
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List

# 상수 정의
QA_PATTERN = re.compile(r'Q\d+:|A\d+:')
INPUT_FILE_PATH = './app/models/medical/fine_tuning/medical_fine_tuning_file.txt'
OUTPUT_FILE_PATH = './app/models/medical/fine_tuning/medical_fine_tuning_file.csv'
FIELDNAMES = ['instruction', 'input', 'output', 'prompt']
MODEL_NAME = "google/gemma-2b"
OUTPUT_DIR = "./app/models/medical/fine_tuning"
MODEL_SAVE_PATH = './app/models/medical/fine_tuning/medical_fine_tuned_gemma'

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV 생성 관련 함수들
def read_file_content(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_qa_pairs(content: str) -> List[str]:
    qa_pairs = QA_PATTERN.split(content)
    return [pair.strip() for pair in qa_pairs if pair.strip()]

def create_prompt(question: str, answer: str) -> str:
    return f"<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n{answer}\n<end_of_turn>"

def process_qa_pairs(qa_pairs: List[str]) -> List[Dict[str, str]]:
    data = []
    for i in range(0, len(qa_pairs), 2):
        if i+1 < len(qa_pairs):
            question = qa_pairs[i]
            answer = qa_pairs[i+1]
            prompt = create_prompt(question, answer)
            data.append({
                'instruction': question,
                'input': '',
                'output': answer,
                'prompt': prompt
            })
    return data

def save_to_csv(data: List[Dict[str, str]], file_path: str) -> None:
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(data)

# 모델 파인튜닝 관련 함수들
def load_env_variables() -> str:
    load_dotenv()
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in .env file")
    return token

def print_trainable_parameters(model: Any) -> None:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    logger.info(
        f"학습 가능한 매개변수: {trainable_params} || 전체 매개변수: {all_param} || "
        f"학습 가능 비율: {100 * trainable_params / all_param:.2f}%"
    )

def load_and_process_dataset(dataset_path: str) -> Dataset:
    dataset = load_dataset('csv', data_files=dataset_path)
    logger.info("데이터셋 컬럼: %s", dataset["train"].column_names)
    logger.info("데이터셋 크기: %d", len(dataset["train"]))
    logger.info("첫 번째 샘플: %s", dataset["train"][0])
    return dataset["train"]

def load_model_and_tokenizer(model_name: str, token: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=token,
    )
    return model, tokenizer

def apply_lora_config(model: Any) -> Any:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)

def preprocess_function(examples: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

def preprocess_dataset(dataset: Dataset, tokenizer: Any) -> Dataset:
    try:
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        logger.info("데이터셋 전처리 완료. 샘플 확인: %s", tokenized_dataset[0])
        return tokenized_dataset
    except Exception as e:
        logger.error(f"데이터셋 전처리 중 오류 발생: {e}")
        raise

def setup_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-6,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        push_to_hub=False,
    )

def main():
    # CSV 파일 생성
    logger.info("CSV 파일 생성 시작")
    content = read_file_content(INPUT_FILE_PATH)
    qa_pairs = split_qa_pairs(content)
    data = process_qa_pairs(qa_pairs)
    save_to_csv(data, OUTPUT_FILE_PATH)
    logger.info(f"데이터가 {OUTPUT_FILE_PATH} 파일로 저장되었습니다.")

    # 모델 파인튜닝
    logger.info("모델 파인튜닝 시작")
    torch.cuda.empty_cache()
    token = load_env_variables()
    
    dataset = load_and_process_dataset(OUTPUT_FILE_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, token)
    model = apply_lora_config(model)
    
    print_trainable_parameters(model)
    
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = setup_training_args()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("학습을 시작합니다.")
    try:
        trainer.train()
        logger.info("학습이 성공적으로 완료되었습니다.")
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}", exc_info=True)
        return
    
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    logger.info(f"파인튜닝이 완료되었습니다. 모델이 '{MODEL_SAVE_PATH}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()