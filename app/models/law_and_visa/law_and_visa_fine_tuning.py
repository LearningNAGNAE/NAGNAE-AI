import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate import init_empty_weights

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

# CSV 파일에서 데이터셋 로드
dataset_path = './app/models/law_and_visa/gemma_fine_tuning_dataset.csv'
dataset = load_dataset('csv', data_files=dataset_path)

# 데이터셋 구조 확인
print("데이터셋 컬럼:", dataset["train"].column_names)
print("첫 번째 예제:", dataset["train"][0])

# 모델과 토크나이저 로드
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 모델을 메타 디바이스에 로드
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name)

# 모델을 실제 weights로 로드
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# LoRA 설정 수정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습 가능한 매개변수 확인
print_trainable_parameters(model)

# 데이터 전처리 함수 수정
def preprocess_function(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# 데이터셋 전처리
try:
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
except Exception as e:
    print(f"데이터셋 전처리 중 오류 발생: {e}")
    print("첫 번째 예제:", dataset["train"][0])
    raise

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 학습 인자 설정 수정
training_args = TrainingArguments(
    output_dir="./app/models/law_and_visa/results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-6,  # 학습률 감소
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=False,  # 그래디언트 체크포인팅 비활성화
    remove_unused_columns=False,
    push_to_hub=False,
)

# Trainer 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 학습 시작
try:
    trainer.train()
    print("학습이 성공적으로 완료되었습니다.")
except Exception as e:
    print(f"학습 중 오류 발생: {e}")
    print("오류 세부 정보:", e.__traceback__)

# 모델 저장
model_save_path = './app/models/law_and_visa/fine_tuned_gemma'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"파인튜닝이 완료되었습니다. 모델이 '{model_save_path}' 디렉토리에 저장되었습니다.")