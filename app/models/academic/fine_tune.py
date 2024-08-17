import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate import init_empty_weights
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
import wandb
from tqdm.auto import tqdm
from transformers import TrainerCallback




# 학습 가능한 매개변수 출력 함수
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"학습 가능한 매개변수: {trainable_params} || 전체 매개변수: {all_param} || 학습 가능 비율: {100 * trainable_params / all_param:.2f}%"
    )

# 1. 데이터 준비
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df

# 임시 RAG 파이프라인 (실제 구현 전까지 사용)
def temp_rag_pipeline(question):
    return f"This is a temporary answer for the question: {question}"

# 2. 훈련 데이터 생성 (RAG 파이프라인 필요)
def generate_training_data(df, rag_pipeline):
    training_data = []
    for _, row in df.iterrows():
        rag_answer = rag_pipeline(row['question'])
        training_data.append({
            'question': row['question'],
            'context': row['document'],
            'rag_answer': rag_answer,
            'true_answer': row['answer']
        })
    return pd.DataFrame(training_data)

# 2. 모델 및 토크나이저 로드
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

# 3. LoRA 설정 및 적용
def apply_lora(model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)

# 4. 데이터 전처리
def preprocess_function(examples, tokenizer):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# 5. 훈련 설정
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=2,  # 배치 사이즈 조정
        gradient_accumulation_steps=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb",
        run_name="gemma_fine_tuning"
    )

# 7. 모델 평가 (간단한 구현)
def evaluate_model(model, test_df, tokenizer):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            input_text = f"Question: {row['question']}\nContext: {row['document']}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(model.device)
            
            outputs = model.generate(**inputs, max_length=100)
            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if predicted_answer.strip().lower() in row['answer'].strip().lower():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"Model Accuracy: {accuracy:.2f}")

class TqdmCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps is not None:
            total = state.max_steps
        else:
            total = state.num_train_epochs * state.num_update_steps_per_epoch
        self.pbar = tqdm(total=total, desc="Training")

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

# 메인 실행 함수
def main():
    # 텍스트 데이터 준비
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'gemma_academic_fine_tuning_dataset.csv')
    train_df, val_df, test_df = prepare_data(file_path)
    print("데이터 준비 완료")

    # 모델 및 토크나이저 로드
    model_name = "google/gemma-2-2b"
    model, tokenizer = load_model_and_tokenizer(model_name)
    print("모델 및 토크나이저 로드 완료")

    # LoRA 적용
    model = apply_lora(model)
    print_trainable_parameters(model)

    # 데이터셋 생성 및 전처리
    dataset = load_dataset('csv', data_files=file_path)
    
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    print("데이터셋 전처리 완료")

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 훈련 인자 설정
    training_args = get_training_args("./app/models/academic/results")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[TqdmCallback()]  # TqdmCallback 인스턴스 생성
    )

    # 학습 시작
    print("학습 시작")
    trainer.train()
    print("학습 완료")

    # 현재 스크립트 파일의 디렉토리 경로를 얻습니다
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 모델 저장 경로를 스크립트 위치를 기준으로 설정합니다
    model_save_path = os.path.join(script_dir, 'fine_tuned_gemma')

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"파인튜닝이 완료되었습니다. 모델이 '{model_save_path}' 디렉토리에 저장되었습니다.")

    # 모델 평가
    evaluate_model(model, test_df, tokenizer)

if __name__ == "__main__":
    main()