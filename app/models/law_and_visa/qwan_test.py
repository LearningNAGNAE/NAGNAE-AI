import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 환경 변수 로드 및 FastAPI 애플리케이션 생성
load_dotenv()
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 데이터 모델 정의
class Query(BaseModel):
    question: str
    session_id: str

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Qwen-2 1.5b 모델 로드
model_name = "qwen/qwen2-1.5b"
tokenizer_qwen = AutoTokenizer.from_pretrained(model_name)
model_qwen = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Qwen 모델을 사용한 응답 생성
def generate_response_with_qwen(question: str) -> str:
    prompt = f"""아래 질문에 대해 자세하고 포괄적인 답변을 제공해주세요. 한국의 비자와 외국인 근로자 및 학생을 위한 한국 법률 정보에 초점을 맞추어 답변해주세요. 관련된 세부 정보와 실용적인 조언을 포함해주세요.

질문: {question}

답변:"""
    
    input_ids = tokenizer_qwen(prompt, return_tensors="pt").to(model_qwen.device)
    with torch.no_grad():
        outputs = model_qwen.generate(
            **input_ids,
            max_new_tokens=256,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=False,
            repetition_penalty=1.2
        )
    response = tokenizer_qwen.decode(outputs[0], skip_special_tokens=True).strip()
    return response

# Qwen을 사용한 질문 분류
def classify_question_with_qwen(question: str) -> str:
    prompt = f"""
    Please classify the question into one of the five categories below:
    - Visa Application
    - Work Permit
    - Student Visa
    - Visa Extension
    - General Living in Korea
    
    Question: {question}
    Category:
    """
    input_ids = tokenizer_qwen(prompt, return_tensors="pt").to(model_qwen.device)
    with torch.no_grad():
        outputs = model_qwen.generate(
            **input_ids,
            max_new_tokens=32,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=False
        )
    category = tokenizer_qwen.decode(outputs[0], skip_special_tokens=True).strip()
    return category

# 채팅 엔드포인트
@app.post("/law")
async def chat_endpoint(query: Query):
    try:
        question = query.question
        session_id = query.session_id
        logger.info(f"Received question: {question}, Session ID: {session_id}")

        # 질문 분류
        category = classify_question_with_qwen(question)
        logger.info(f"Question category: {category}")

        # Qwen을 사용한 응답 생성
        qwen_response = generate_response_with_qwen(question)

        return {
            "question": question,
            "category": category,
            "answer": qwen_response,
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)