import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
import pdfplumber
import re
from spacy import load
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tiktoken
from functools import lru_cache

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

def extract_pdf_content(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_content = []
        for page in pdf.pages:
            all_content.append(page.extract_text())
            for table in page.extract_tables():
                processed_table = [[str(cell) if cell is not None else '' for cell in row] for row in table]
                all_content.append("\n".join(["\t".join(row) for row in processed_table]))
    return "\n.join(all_content)"

# PDF 내용 추출 및 처리
pdf_content = extract_pdf_content(pdf_path)

# FAISS 인덱스 설정
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_texts([pdf_content], embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# BM25 검색기 설정
bm25_retriever = BM25Retriever.from_texts([pdf_content], k=2)

# 하이브리드 검색기 설정
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# TF-IDF 벡터라이저 초기화 및 피팅
vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = vectorizer.fit_transform([pdf_content])
feature_names = vectorizer.get_feature_names_out()

# 키워드 인덱스 생성
keyword_index = {word: [] for word in feature_names}
for i, word in enumerate(feature_names):
    if tfidf_matrix[0, i] > 0:
        keyword_index[word].append((tfidf_matrix[0, i], i))

# 키워드 검색 함수
def search_keywords(query, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = []
    for word in query_vec.indices:
        if word < len(feature_names):
            word_text = feature_names[word]
            if word_text in keyword_index:
                scores.extend(keyword_index[word_text])
    return sorted(scores, reverse=True)[:top_k]


# ChatOpenAI 모델 초기화
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# NER 모델 로드
nlp = load("ko_core_news_sm")

# System prompt
system_prompt = """
AI Assistant for University Admission Policies for International Students in Korea
Role and Responsibility
You are a specialized AI assistant focused on analyzing and providing information about university admission policies for international students in Korea. Your primary goals are to:

1. Provide Accurate Information:
    - Offer precise, detailed, and easily understandable information based on the PDF documents you have access to.
    - Always verify and cross-check information from multiple sources when possible.

2. Address Specific Queries:
    - Address both general and specific questions related to university admission policies, ensuring a focus on information relevant to international students.
    - Provide examples or hypothetical scenarios when applicable to clarify policies and procedures.
    - If a question includes a specific region, provide information about all universities in that region from the provided PDF documents and `region_translations`.

3. Ensure Clarity and Precision:
    - Guide users clearly on admission requirements, procedures, and related matters.
    - Use bullet points or numbered lists to break down complex information.
    - Highlight specific deadlines, steps, and potential consequences of non-compliance.

4. Be Culturally Sensitive:
    - Maintain awareness of cultural differences and address queries with sensitivity.
    - Adapt communication style to be respectful and considerate of diverse cultural backgrounds.

Guidelines
Language:
    - ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the user's question language.

Information Scope:
    - Admission Policies: Provide detailed information on application requirements, procedures, deadlines, and any specific criteria for international students.
    - Documents Required: Explain the types of documents needed, such as proof of previous academic qualifications, language proficiency test scores, and others.
    - Special Admission Procedures: Detail any special procedures or additional requirements for overseas Koreans or foreign students.
    - Scholarships and Financial Aid: Offer guidance on available scholarships, eligibility, and application procedures.
    - Health Insurance and Housing: Provide insights into university policies on health insurance, accommodation, and other living conditions.

Specific Focus Areas:
    - Document Requirements: Provide clear details on the necessary documentation for different types of admissions.
    - Application Procedures: Explain the step-by-step process for applying, including any special procedures.
    - Deadlines and Time Limits: Clearly outline application deadlines and time limits for submission or changes.
    - Consequences of Non-Compliance: Detail the implications of failing to meet admission requirements or deadlines.

Completeness:
    - Ensure your answers are comprehensive. Include specific deadlines and time limits, required steps and procedures, and potential consequences of not following the guidelines.
    - Consider variations based on specific circumstances, if applicable.

Accuracy and Updates:
    - Emphasize that while you provide detailed information based on the documents, policies may change. Always advise users to verify current details with official sources.

Handling Uncertainty:
    - If uncertain about specific details, clearly state this and provide the most relevant general information available.
    - Recommend consulting official sources for the most accurate and case-specific guidance.

Regional Information:
    - When a question includes a specific region, ensure to provide information about all universities in that region from the provided PDF documents and `region_translations`.
    - The regions to be recognized include Seoul, Busan, Daegu, Incheon, Gwangju, Daejeon, Ulsan, Gyeonggi-do, Gangwon-do, Chungcheongbuk-do, Chungcheongnam-do, Jeollabuk-do, Jeollanam-do, Gyeongsangbuk-do, Gyeongsangnam-do, and Jeju-do.
    - Make sure to clearly distinguish between these regions and provide relevant information accordingly.
    - Do not include universities that are not listed in the PDF documents.

Remember, your role is to deliver relevant, accurate, and detailed information in a clear and actionable manner for the user.

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

# 토큰 제한 함수
def limit_tokens(text, max_tokens=2000):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

# 검색 체인 구성
def get_context(question: str):
    # 키워드 기반 검색
    top_keywords = search_keywords(question, top_k=2)
    keyword_context = "\n".join([pdf_content[kw[1]:kw[1]+300] for kw in top_keywords])
    
    # FAISS 및 BM25 검색
    ensemble_docs = ensemble_retriever.get_relevant_documents(question)
    ensemble_context = "\n".join(doc.page_content[:300] for doc in ensemble_docs[:1])
    
    # NER을 사용하여 질문에서 지역 엔티티 추출
    doc = nlp(question)
    mentioned_regions = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    
    if mentioned_regions:
        region_docs = vectorstore.similarity_search(question, k=1, filter={"region": mentioned_regions[0]})
        region_context = "\n".join(doc.page_content[:300] for doc in region_docs)
    else:
        region_context = ""
    
    # 결과 조합 및 토큰 제한
    combined_context = f"{keyword_context}\n\n{ensemble_context}\n\n{region_context}"
    limited_context = limit_tokens(combined_context, max_tokens=2000)
    
    return post_process_context(limited_context, question)

def post_process_context(context, question):
    regions = ["충북", "충남", "경북", "경남", "전북", "전남", "강원도", "제주도", "서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기도"]
    mentioned_region = next((region for region in regions if region in question), None)
    
    if mentioned_region:
        lines = context.split('\n')
        relevant_lines = [line for line in lines if mentioned_region in line]
        return "\n".join(relevant_lines) if relevant_lines else context
    return context

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

# 지역명 번역 데이터
region_translations = {
    "서울": {
        "ko": "서울", "zh": "首尔", "ja": "ソウル", "vi": "Seoul", "uz": "Seul", "mn": "Сөүл",
        "full_name": {"ko": "서울특별시", "zh": "首尔特别市", "ja": "ソウル特別市", "vi": "Thành phố Seoul", "uz": "Seul shahar", "mn": "Сөүл"}
    },
    "부산": {
        "ko": "부산", "zh": "釜山", "ja": "釜山", "vi": "Busan", "uz": "Busan", "mn": "Пусан",
        "full_name": {"ko": "부산광역시", "zh": "釜山广域市", "ja": "釜山広域市", "vi": "Thành phố Busan", "uz": "Busan shahar", "mn": "Пусан"}
    },
    "대구": {
        "ko": "대구", "zh": "大邱", "ja": "大邱", "vi": "Daegu", "uz": "Daegu", "mn": "대구",
        "full_name": {"ko": "대구광역시", "zh": "大邱广域市", "ja": "大邱広域市", "vi": "Thành phố Daegu", "uz": "Daegu shahar", "mn": "대구"}
    },
    "인천": {
        "ko": "인천", "zh": "仁川", "ja": "仁川", "vi": "Incheon", "uz": "Incheon", "mn": "인천",
        "full_name": {"ko": "인천광역시", "zh": "仁川广域市", "ja": "仁川広域市", "vi": "Thành phố Incheon", "uz": "Incheon shahar", "mn": "인천"}
    },
    "광주": {
        "ko": "광주", "zh": "光州", "ja": "光州", "vi": "Gwangju", "uz": "Gwangju", "mn": "광주",
        "full_name": {"ko": "광주광역시", "zh": "光州广域市", "ja": "光州広域市", "vi": "Thành phố Gwangju", "uz": "Gwangju shahar", "mn": "광주"}
    },
    "대전": {
        "ko": "대전", "zh": "大田", "ja": "大田", "vi": "Daejeon", "uz": "Daejeon", "mn": "대전",
        "full_name": {"ko": "대전광역시", "zh": "大田广域市", "ja": "大田広域市", "vi": "Thành phố Daejeon", "uz": "Daejeon shahar", "mn": "대전"}
    },
    "울산": {
        "ko": "울산", "zh": "蔚山", "ja": "蔚山", "vi": "Ulsan", "uz": "Ulsan", "mn": "울산",
        "full_name": {"ko": "울산광역시", "zh": "蔚山广域市", "ja": "蔚山広域市", "vi": "Thành phố Ulsan", "uz": "Ulsan shahar", "mn": "울산"}
    },
    "경기도": {
        "ko": "경기도", "zh": "京畿道", "ja": "京畿道", "vi": "Gyeonggi-do", "uz": "Gyeonggi-do", "mn": "Кёнгидо",
        "full_name": {"ko": "경기도", "zh": "京畿道", "ja": "京畿道", "vi": "Tỉnh Gyeonggi", "uz": "Gyeonggi-do", "mn": "Кёнгидо"}
    },
    "충북": {
        "ko": "충북", "zh": "忠北", "ja": "忠北", "vi": "Chungbuk", "uz": "Chungbuk", "mn": "Чунбук",
        "full_name": {"ko": "충청북도", "zh": "忠清北道", "ja": "忠清北道", "vi": "Tỉnh Chungbuk", "uz": "Chungbuk", "mn": "Чунбук"}
    },
    "충남": {
        "ko": "충남", "zh": "忠南", "ja": "忠南", "vi": "Chungnam", "uz": "Chungnam", "mn": "Чуннам",
        "full_name": {"ko": "충청남도", "zh": "忠清南道", "ja": "忠清南道", "vi": "Tỉnh Chungnam", "uz": "Chungnam", "mn": "Чуннам"}
    },
    "경남": {
        "ko": "경남", "zh": "庆南", "ja": "慶南", "vi": "Gyeongnam", "uz": "Gyeongnam", "mn": "Гёнсан",
        "full_name": {"ko": "경상남도", "zh": "庆尚南道", "ja": "慶尚南道", "vi": "Tỉnh Gyeongnam", "uz": "Gyeongnam", "mn": "Гёнсан"}
    },
    "경북": {
        "ko": "경북", "zh": "庆北", "ja": "慶北", "vi": "Gyeongbuk", "uz": "Gyeongbuk", "mn": "Гёнбук",
        "full_name": {"ko": "경상북도", "zh": "庆尚北道", "ja": "慶尚北道", "vi": "Tỉnh Gyeongbuk", "uz": "Gyeongbuk", "mn": "Гёнбук"}
    },
    "전북": {
        "ko": "전북", "zh": "全罗北道", "ja": "全羅北道", "vi": "Jeollabuk-do", "uz": "Jeollabuk-do", "mn": "전라북도",
        "full_name": {"ko": "전라북도", "zh": "全罗北道", "ja": "全羅北道", "vi": "Tỉnh Jeollabuk", "uz": "Jeollabuk-do", "mn": "전라북도"}
    },
    "전남": {
        "ko": "전남", "zh": "全罗南道", "ja": "全羅南道", "vi": "Jeollanam-do", "uz": "Jeollanam-do", "mn": "전라남도",
        "full_name": {"ko": "전라남도", "zh": "全罗南道", "ja": "全羅南道", "vi": "Tỉnh Jeollanam", "uz": "Jeollanam-do", "mn": "전라남도"}
    },
    "강원도": {
        "ko": "강원도", "zh": "江原道", "ja": "江原道", "vi": "Gangwon-do", "uz": "Gangwon-do", "mn": "강원도",
        "full_name": {"ko": "강원도", "zh": "江原道", "ja": "江原道", "vi": "Tỉnh Gangwon", "uz": "Gangwon-do", "mn": "강원도"}
    },
    "제주도": {
        "ko": "제주도", "zh": "济州岛", "ja": "済州島", "vi": "Jeju-do", "uz": "Jeju-do", "mn": "제주도",
        "full_name": {"ko": "제주특별자치도", "zh": "济州特别自治道", "ja": "済州特別自治道", "vi": "Tỉnh Jeju", "uz": "Jeju-do", "mn": "제주특별자치도"}
    }
}

# 캐시 설정
@lru_cache(maxsize=100)
def cached_chat_response(question: str, language: str, session_id: str):
    response = chain_with_history.invoke(
        {"question": question, "language": language},
        {"configurable": {"session_id": session_id}}
    )
    return response if isinstance(response, str) else "Error: Unable to generate response"


# 채팅 엔드포인트
@app.post("/study")
async def chat_endpoint(query: Query):
    try:
        question = query.question
        session_id = query.session_id

        language = detect_language(question)

        # 캐시된 응답 사용
        answer = cached_chat_response(question, language, session_id)

        # 요청 간 지연 추가
        time.sleep(1)

        return {
            "question": question,
            "answer": answer,
            "detected_language": language,
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        if "rate_limit_exceeded" in str(e):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# 지역명 번역 엔드포인트
@app.get("/translate_region")
async def translate_region_endpoint(region_name: str, language_code: str):
    translations = region_translations.get(region_name)
    if translations:
        translated_name = translations.get(language_code)
        full_name = translations.get("full_name", {}).get(language_code)
        return {
            "region_name": region_name,
            "language_code": language_code,
            "translated_name": translated_name,
            "full_name": full_name
        }
    else:
        return {"error": "Translation not found for the given region and language."}

# 언어 감지 함수
@lru_cache(maxsize=100)
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
