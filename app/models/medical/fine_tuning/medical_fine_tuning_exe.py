from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
question = "Can international students also enroll in National Health Insurance?"
print(f"Using device: {device}")


# 프롬프트
prompt = f"""
Health advice for foreigners in Korea: {question}
- Give clear, accurate info
- Explain relevant services
- Use simple language
- Advise when to see a doctor
"""

# Gemma 2-2b 모델 
# gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# gemaa_model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2b",
#     device_map="auto",
#     torch_dtype=torch.float16
# )

# gemma_inputs = gemma_tokenizer(prompt.format(question=question), return_tensors="pt").to(device)

# gemma_response = gemaa_model.generate(
#     **gemma_inputs,
#     max_new_tokens=200,
#     num_return_sequences=1,
#     temperature=0.0,
#     top_k=50,
#     top_p=0.95,
#     no_repeat_ngram_size=2,
#     early_stopping=True
# )

# fine-tuning 모델
model_path = './app/models/medical/fine_tuning/medical_fine_tuned_gemma'
fine_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
fine_inputs = fine_tokenizer(prompt.format(question=question), return_tensors="pt").to(device)
fine_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype=torch.float16
        )

fine_inputs = fine_tokenizer(prompt.format(question=question), return_tensors="pt").to(device)
fine_gemma_response = fine_model.generate(
    **fine_inputs,
    max_new_tokens=200,
    num_return_sequences=1,
    temperature=0.0,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    early_stopping=True
)


# 답변 생성
# gemma_response_result = gemma_tokenizer.decode(gemma_response[0][gemma_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
fine_gemma_response_result = fine_tokenizer.decode(fine_gemma_response[0][fine_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
# 결과 출력
# print(f"[Gemma 2-2b 답변]:\n{gemma_response_result.strip()}")
print(f"[fine Gemma 2-2b 답변]:\n{fine_gemma_response_result.strip()}")