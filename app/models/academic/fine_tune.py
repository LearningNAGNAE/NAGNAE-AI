import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate import init_empty_weights
from sklearn.model_selection import train_test_split
import pandas as pd

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
def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
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

# 3. 모델 및 토크나이저 로드
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

# 4. LoRA 설정 및 적용
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

# 5. 데이터 전처리
def preprocess_function(examples, tokenizer):
    prompts = examples["prompt"]
    # 'prompt'가 리스트가 아니라면 리스트로 변환
    if not isinstance(prompts, list):
        prompts = [prompts]
    
    # None 값을 빈 문자열로 대체
    prompts = [str(p) if p is not None else "" for p in prompts]
    
    inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# 6. 훈련 설정
def get_training_args(output_dir):
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=1e-4,  # 이 값을 조정해보세요 (예: 1e-4 또는 1e-5)
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    args.lr_scheduler_type = 'linear'
    args.warmup_ratio = 0.1
    return args

# 7. 모델 평가 (간단한 구현)
def evaluate_model(model, test_dataset, tokenizer):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in test_dataset:
            input_text = item['prompt']
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(model.device)
            
            outputs = model.generate(**inputs, max_length=100)
            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 'output' 열의 내용과 비교
            if predicted_answer.strip().lower() in item['output'].strip().lower():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"Model Accuracy: {accuracy:.2f}")


def prepare_dataset(dataset):
    if 'prompt' not in dataset.column_names:
        dataset = dataset.map(lambda example: {
            'prompt': f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
        })
    return dataset

# 메인 실행 함수
def main():
    # 데이터 준비
    csv_path = './app/models/academic/gemma_academic_fine_tuning_dataset.csv'
    dataset = load_dataset('csv', data_files=csv_path)
    dataset = prepare_dataset(dataset['train'])  # 'train' 키를 사용

    # 데이터셋 분할
    dataset = dataset.train_test_split(test_size=0.2)


    print("데이터 준비 완료")

    # 모델 및 토크나이저 로드
    model_name = "qwen/qwen2-1.5b"
    model, tokenizer = load_model_and_tokenizer(model_name)
    print("모델 및 토크나이저 로드 완료")

    # LoRA 적용
    model = apply_lora(model)
    print_trainable_parameters(model)

    # 데이터셋 전처리
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

    # Trainer 초기화 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 학습 시작
    print("학습 시작")
    trainer.train()
    print("학습 완료")

    # 훈련 후 메모리 해제
    torch.cuda.empty_cache()

    # 모델 저장
    model_save_path = './app/models/academic/fine_tuned_qwen2_1_5b'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"파인튜닝이 완료되었습니다. 모델이 '{model_save_path}' 디렉토리에 저장되었습니다.")

    # 모델 평가
    evaluate_model(model, tokenized_dataset["test"], tokenizer)

    # 평가 후 메모리 해제
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
