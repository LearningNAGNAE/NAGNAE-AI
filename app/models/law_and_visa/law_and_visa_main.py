import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseChatMessageHistory
import logging
import traceback

# 환경 변수 로드 및 FastAPI 애플리케이션 생성
load_dotenv()
app = FastAPI()

# 입력 데이터 모델 정의
class Query(BaseModel):
    question: str
    session_id: str

# FAISS 인덱스 로드 및 설정
embeddings = OpenAIEmbeddings()
index_name = "faiss_index_law_and_visa_page"
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, index_name)

# FAISS 인덱스가 존재하지 않으면 예외 발생
if not os.path.exists(index_path):
    raise HTTPException(status_code=500, detail=f"FAISS index not found: {index_path}")

vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# BM25 인덱스 로드 및 설정
bm25_texts = [
    "Korean labor laws protect workers' rights to fair wages and working conditions.",
    "The visa application process in Korea involves several steps and documentation.",
    # 추가 텍스트 데이터 ...
]  # BM25 인덱스에 사용할 문서 리스트를 여기에 설정
bm25_retriever = BM25Retriever.from_texts(bm25_texts, k=4)

# 하이브리드 검색기 설정
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

print(f"FAISS index size: {vector_store.index.ntotal}")

# 벡터 저장소를 검색기로 사용
retriever = ensemble_retriever

# 시스템 프롬프트 
system_prompt = """
# AI Assistant for Foreign Workers and Students in Korea: Legal and General Information

## Role and Responsibility
You are a specialized AI assistant providing information on Korean law, visa regulations, labor rights, and general topics for foreign workers and students in Korea. Your primary goals are to:

1. Provide accurate, easy-to-understand information in the language specified in the 'RESPONSE_LANGUAGE' field.
2. Offer both general and specific information relevant to foreign workers and students.
3. Guide users on labor rights, visa regulations, and academic-related legal matters.
4. Ensure cultural sensitivity and awareness in all interactions.
5. Provide relevant contact information or website addresses for official authorities when applicable.

## Guidelines

1. Language: ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the user's question language.

2. Information Scope:
   - Labor Laws: Work hours, wages, benefits, worker protection
   - Visa Regulations: Types, application processes, restrictions, changes
   - Academic Laws: Student rights, academic integrity, scholarship regulations
   - General Living: Healthcare, housing, transportation, cultural norms

3. Specific Focus Areas:
   - Clear distinction between general rules and specific visa/worker type regulations
   - Guidance on worker's and student's rights
   - Information on dispute resolution and official channels for help

4. Cultural Sensitivity: Be aware of cultural differences and provide necessary context.

5. Uncertainty Handling: If uncertain, respond in the specified language with:
   "Based on the provided information, I cannot give a definitive answer. Please consult [relevant authority] or a legal expert specializing in [specific area] for accurate advice."

6. Legal Disclaimers: Emphasize that laws may change and encourage verifying current regulations with official sources.

7. Contact Information: When relevant, provide official contact information or website addresses for appropriate government agencies or organizations. For example:
   - Ministry of Justice (출입국·외국인정책본부): www.immigration.go.kr
   - Ministry of Employment and Labor: www.moel.go.kr
   - Korea Immigration Service: www.hikorea.go.kr
   - National Health Insurance Service: www.nhis.or.kr

Always approach each query systematically to ensure accurate, helpful, and responsible assistance. Prioritize the well-being and legal protection of foreign workers and students in your responses, and guide them to official sources when necessary.
"""

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프롬프트 템플릿 설정
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

human_template = """
RESPONSE_LANGUAGE: {language}
CONTEXT: {context}
QUESTION: {question}

Please provide a detailed answer to the above question in the specified RESPONSE_LANGUAGE.
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

ai_message_prompt = AIMessagePromptTemplate.from_template("answer:")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
    ai_message_prompt
])

# ChatOpenAI 모델 초기화
model = ChatOpenAI(temperature=0)

# 검색 체인 구성
retrieval_chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"],
        "language": lambda x: x["language"]
    }
    | chat_prompt
    | model
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
def detect_language(text):
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = chat.invoke(messages)
    detected_language = response.content.strip().lower()
    logger.debug(f"Detected language: {detected_language}")
    return detected_language

# 요청 유효성 검사 예외 처리기
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

# CORS 미들웨어 추가 (임시)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 채팅 엔드포인트 수정
@app.post("/law")
async def chat_endpoint(query: Query):
    try:
        question = query.question if isinstance(query.question, str) else str(query.question)
        session_id = query.session_id
        logger.debug(f"Received question: {question}")

        # 언어 감지
        language = detect_language(question)
        logger.debug(f"Detected language: {language}")

        if retriever is None:
            error_message = "System error. Please try again later."
            return {
                "question": question,
                "answer": error_message,
                "detected_language": language,
            }

        # RunnableWithMessageHistory를 사용하여 응답 생성
        response = chain_with_history.invoke(
            {
                "question": {"question": question, "language": language},
            },
            {"configurable": {"session_id": session_id}}
        )
        answer = response.get("answer", "")
        logger.debug(f"Generated answer: {answer}")

        return {
            "question": question,
            "answer": answer,
            "detected_language": language,
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error"}
        )

# 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
   
