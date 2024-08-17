import os
import torch
import logging
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate import init_empty_weights
from dotenv import load_dotenv
from typing import Dict, Any
from transformers import TrainerCallback

# 상수 정의
MODEL_NAME = "google/gemma-2b"
OUTPUT_DIR = "./app/models/medical/fine_tuning"
MODEL_SAVE_PATH = './app/models/medical/fine_tuning/medical_fine_tuned_gemma'
OUTPUT_FILE_PATH = './app/models/medical/fine_tuning/medical_fine_tuning_file.csv'

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizationCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, num_examples=5):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_examples = num_examples

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n=== 학습 데이터 샘플 ===")
        for i in range(min(self.num_examples, len(self.dataset))):
            example = self.dataset[i]
            print(f"예시 {i+1}:")
            print("입력:", self.tokenizer.decode(example['input_ids']))
            print("라벨:", self.tokenizer.decode(example['labels']))
            print()

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
        r=16,
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
        num_train_epochs=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        push_to_hub=False,
        logging_dir='./logs',  # 로그를 저장할 디렉토리
        logging_strategy="steps",  # 각 스텝마다 로깅
        logging_first_step=True,  # 첫 번째 스텝에서도 로깅
        report_to=["tensorboard"]  # TensorBoard를 사용하여 로그 시각화
    )

def main():
    logger.info("모델 파인튜닝 시작")
    torch.cuda.empty_cache()
    token = load_env_variables()
    
    dataset = load_and_process_dataset(OUTPUT_FILE_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, token)
    model = apply_lora_config(model)

    # for param in model.parameters():
    #     param.requires_grad = True

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
        callbacks=[DataVisualizationCallback(tokenizer, tokenized_dataset)]
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
