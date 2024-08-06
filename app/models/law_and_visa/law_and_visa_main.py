import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# 메모리 사용량 최적화
torch.cuda.empty_cache()

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

# Fine-tuned Gemma-2b 모델 및 토크나이저 로드
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_path = './app/models/law_and_visa/fine_tuned_gemma'  # Fine-tuned 모델이 저장된 경로
tokenizer_gemma = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model_gemma = AutoModelForCausalLM.from_pretrained(
    model_path, 
    quantization_config=quantization_config, 
    local_files_only=True,
    device_map="auto"  # 이 옵션을 추가하여 자동으로 적절한 디바이스에 모델을 로드합니다.
)

# FAISS 인덱스 로드 및 설정
embeddings = OpenAIEmbeddings()
index_name = "faiss_index_law_and_visa_page"
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, index_name)

if not os.path.exists(index_path):
    raise HTTPException(status_code=500, detail=f"FAISS index not found: {index_path}")

vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# BM25 및 FAISS 검색기 설정
bm25_texts = [
    "For all visa types: Overstaying can result in fines, deportation, and future entry bans. Always consult the Korea Immigration Service for the most up-to-date information.",
    "Work permit is separate from visa and may need to be updated when changing jobs, even if visa is still valid.",
    "Some visas require a certain salary level or job position to be maintained. Changing to a lower-paying job may affect visa status.",
    "Health insurance and pension requirements may vary depending on visa type and employment status.",
    "For visa inquiries, contact the Korea Immigration Service at 1345 or visit www.immigration.go.kr",
]
bm25_retriever = BM25Retriever.from_texts(bm25_texts, k=4)

# 하이브리드 검색기 설정
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.3, 0.7]
)

# ChatOpenAI 모델 초기화
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# System prompt
system_prompt = """
# AI Assistant for Foreign Workers and Students in Korea: Detailed Legal and Visa Information

## Role and Responsibility
You are a specialized AI assistant providing comprehensive information on Korean law, visa regulations, labor rights, and general topics for foreign workers and students in Korea. Your primary goals are to:

1. Provide accurate, detailed, and easy-to-understand information in the language specified in the 'RESPONSE_LANGUAGE' field.
2. Offer both general and specific information relevant to foreign workers and students, with a focus on visa-related queries.
3. Guide users on labor rights, visa regulations, and academic-related legal matters with precision.
4. Ensure cultural sensitivity and awareness in all interactions.

## Guidelines

1. Language: ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the user's question language.

2. Information Scope:
   - Visa Regulations: Provide detailed information on different visa types, application processes, restrictions, changes, and time limits for job changes or leaving the country.
   - Labor Laws: Explain work hours, wages, benefits, worker protection, and how they may vary by visa type.
   - Academic Laws: Detail student rights, academic integrity, scholarship regulations, and how they interact with visa status.
   - General Living: Offer insights on healthcare, housing, transportation, and cultural norms as they relate to visa holders.

3. Specific Focus Areas:
   - Provide clear distinctions between rules for different visa types (e.g., E-7, E-9, D-10, F-2-7, F-4).
   - Offer guidance on worker's and student's rights specific to their visa category.
   - Explain the process and time limits for changing jobs or extending stay for each relevant visa type.
   - Detail the consequences of overstaying or violating visa conditions.

4. Completeness: Always provide a comprehensive answer based on the available context. Include:
   - Specific time limits or deadlines
   - Required procedures (e.g., reporting to immigration, applying for changes)
   - Potential consequences of non-compliance
   - Variations based on specific circumstances (if known)

5. Accuracy and Updates: Emphasize that while you provide detailed information, laws and regulations may change. Always advise users to verify current rules with official sources.

6. Structured Responses: Organize your responses clearly, using bullet points or numbered lists when appropriate to break down complex information.

7. Examples and Scenarios: When relevant, provide examples or hypothetical scenarios to illustrate how rules apply in practice.

8. Uncertainty Handling: If uncertain about specific details, clearly state this and provide the most relevant general information available. Always recommend consulting official sources for the most up-to-date and case-specific guidance.

Remember, your goal is to provide as much relevant, accurate, and detailed information as possible while ensuring it's understandable and actionable for the user.
"""

# 프롬프트 템플릿 설정
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

human_template = """
RESPONSE_LANGUAGE: {language}
CONTEXT: {context}
QUESTION: {question}
GEMMA_RESPONSE: {gemma_response}

Please provide a detailed and comprehensive answer to the above question in the specified RESPONSE_LANGUAGE, including specific visa information when relevant. Incorporate insights from the GEMMA_RESPONSE if applicable. Organize your response clearly and include all pertinent details.
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
])

# 검색 체인 구성
def get_context(question: str):
    docs = ensemble_retriever.get_relevant_documents(question)
    context = "\n".join(doc.page_content for doc in docs)
    return process_context(context)

# Gemma-2b로 텍스트 생성 함수
def generate_text_with_gemma(question: str) -> str:
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    input_ids = tokenizer_gemma(prompt, return_tensors="pt").to(model_gemma.device)
    outputs = model_gemma.generate(**input_ids, max_length=512, num_return_sequences=1)
    return tokenizer_gemma.decode(outputs[0], skip_special_tokens=True)

retrieval_chain = (
    {
        "context": lambda x: get_context(x["question"]),
        "question": lambda x: x["question"],
        "language": lambda x: x["language"],
        "gemma_response": lambda x: generate_text_with_gemma(x["question"])
    }
    | chat_prompt
    | chat
    | StrOutputParser()
)

# 메모리 저장소 설정
memory_store = {}

def get_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# RunnableWithMessageHistory 설정
chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# 언어 감지 함수 정의
def detect_language(text: str) -> str:
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = chat.invoke(messages)
    detected_language = response.content.strip().lower()
    logger.info(f"Detected language: {detected_language}")
    return detected_language

# 컨텍스트 처리 함수 추가
def process_context(context: str) -> dict:
    info = {
        "general": [],
        "specific": [],
        "disclaimer": "For the most up-to-date and accurate information, please consult the official Korea Immigration Service website or speak with an immigration officer."
    }
    
    lines = context.split('\n')
    for line in lines:
        if "typically" in line.lower() or "generally" in line.lower():
            info["general"].append(line.strip())
        elif "specific" in line.lower() or "exact" in line.lower():
            info["specific"].append(line.strip())
    
    return info

# 채팅 엔드포인트
@app.post("/law")
async def chat_endpoint(query: Query):
    try:
        question = query.question
        session_id = query.session_id
        logger.info(f"Received question: {question}")

        language = detect_language(question)
        
        response = await chain_with_history.ainvoke(
            {"question": question, "language": language},
            config={"configurable": {"session_id": session_id}},
        )

        return {
            "question": question,
            "answer": response,
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)