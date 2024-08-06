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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# PDF 파일 경로 설정
pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\2025학년도 재외국민과 외국인 특별전형 시행계획 주요사항.pdf"

# PDF 로드 및 처리
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# FAISS 인덱스 설정
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_splits, embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# BM25 검색기 설정
bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = 5

# 하이브리드 검색기 설정
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.3, 0.7]
)

# ChatOpenAI 모델 초기화
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

# System prompt
system_prompt = """
You are an AI assistant specialized in analyzing and answering questions about university admission policies for international students in Korea. 
Your primary task is to provide accurate and helpful information based on the PDF documents you have access to.
When answering questions:
1. Always refer to the information in the PDF documents.
2. If you're not sure about something, say so rather than guessing.
3. Provide specific details and cite the relevant sections of the document when possible.
4. If a question is outside the scope of the information in the documents, politely inform the user.
5. Be concise in your responses, but provide enough detail to be helpful.
6. Respond in the same language as the input question. If the input is in English, respond in English. If it's in Korean, respond in Korean. For other languages, respond in the same language as the input.
"""

# 프롬프트 템플릿 설정
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

human_template = """
RESPONSE_LANGUAGE: {language}
CONTEXT: {context}
QUESTION: {question}

Please provide a detailed and comprehensive answer to the above question in the specified RESPONSE_LANGUAGE, including specific admission policy information when relevant. Organize your response clearly and include all pertinent details.
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
])

# 검색 체인 구성
def get_context(question: str):
    docs = ensemble_retriever.get_relevant_documents(question)
    return "\n".join(doc.page_content for doc in docs)

retrieval_chain = (
    {
        "context": lambda x: get_context(x["question"]),
        "question": lambda x: x["question"],
        "language": lambda x: x["language"]
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

# 채팅 엔드포인트
@app.post("/study")
async def chat_endpoint(query: Query):
    try:
        question = query.question
        session_id = query.session_id
        logger.info(f"Received question: {question}")

        language = detect_language(question)
        logger.info(f"Detected language: {language}")

        # Generate response using RunnableWithMessageHistory
        response = chain_with_history.invoke(
            {"question": question, "language": language},
            {"configurable": {"session_id": session_id}}
        )
        answer = response if isinstance(response, str) else "Error: Unable to generate response"

        return {
            "question": question,
            "answer": answer,
            "detected_language": language,
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)