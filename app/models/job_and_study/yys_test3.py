from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import os
from dotenv import load_dotenv
import re
from konlpy.tag import Kkma
from typing import List, Dict
from langchain.retrievers import BM25Retriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from contextlib import asynccontextmanager

load_dotenv()

app = FastAPI()

class Query(BaseModel):
    input: str
    session_id: str

pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\2025학년도 재외국민과 외국인 특별전형 시행계획 주요사항.pdf"

# 세션별 메모리 저장소
session_memories = {}

# KoNLPy 초기화
kkma = Kkma()

# 지역명 사전 정의
region_dict = {
    "서울": ["서울", "서울시", "서울특별시"],
    "경기": ["경기", "경기도"],
    "인천": ["인천", "인천시", "인천광역시"],
    "부산": ["부산", "부산시", "부산광역시"],
    "대구": ["대구", "대구시", "대구광역시"],
    "광주": ["광주", "광주시", "광주광역시"],
    "대전": ["대전", "대전시", "대전광역시"],
    "울산": ["울산", "울산시", "울산광역시"],
    "세종": ["세종", "세종시", "세종특별자치시"],
    "강원": ["강원", "강원도"],
    "충북": ["충북", "충청북도"],
    "충남": ["충남", "충청남도"],
    "전북": ["전북", "전라북도"],
    "전남": ["전남", "전라남도"],
    "경북": ["경북", "경상북도"],
    "경남": ["경남", "경상남도"],
    "제주": ["제주", "제주도", "제주특별자치도"]
}

# 지역별 대학교 정보 (예시, 실제 데이터로 채워넣어야 함)
university_by_region = {
    "서울": [
        "서울대학교", "고려대학교", "연세대학교", "서강대학교", "성균관대학교", "한양대학교", 
        "중앙대학교", "경희대학교", "홍익대학교", "동국대학교", "건국대학교", "숙명여자대학교", 
        "이화여자대학교", "한국외국어대학교", "서울시립대학교", "숭실대학교", "세종대학교", 
        "국민대학교", "덕성여자대학교", "동덕여자대학교", "서울과학기술대학교", "삼육대학교", 
        "상명대학교", "성신여자대학교", "한성대학교", "KC대학교", "감리교신학대학교", 
        "서울기독대학교", "서울장신대학교", "성공회대학교", "총신대학교", "추계예술대학교", 
        "한국성서대학교", "한국체육대학교", "한영신학대학교"
    ],
    "경기": [
        "아주대학교", "성균관대학교(자연과학캠퍼스)", "한국외국어대학교(글로벌캠퍼스)", "경희대학교(국제캠퍼스)", 
        "가천대학교", "경기대학교", "단국대학교", "한양대학교(ERICA)", "명지대학교", "강남대학교", 
        "경동대학교", "수원대학교", "신한대학교", "안양대학교", "용인대학교", "을지대학교", 
        "평택대학교", "한경대학교", "한국산업기술대학교", "한국항공대학교", "한세대학교", 
        "협성대학교", "가톨릭대학교", "루터대학교", "서울신학대학교", "성결대학교", 
        "중앙승가대학교", "칼빈대학교"
    ],
    "인천": [
        "인천대학교", "인하대학교", "가천대학교(메디컬캠퍼스)", "경인교육대학교", "인천가톨릭대학교"
    ],
    "부산": [
        "부산대학교", "동아대학교", "부경대학교", "동의대학교", "경성대학교", "신라대학교", 
        "고신대학교", "부산외국어대학교", "동서대학교", "한국해양대학교", "부산가톨릭대학교"
    ],
    "대구": [
        "경북대학교", "계명대학교", "영남대학교", "대구대학교", "대구가톨릭대학교", 
        "대구한의대학교", "금오공과대학교", "경일대학교", "대구예술대학교"
    ],
    "광주": [
        "전남대학교", "조선대학교", "광주과학기술원", "호남대학교", "광주대학교", 
        "광주여자대학교", "남부대학교", "송원대학교"
    ],
    "대전": [
        "충남대학교", "한국과학기술원(KAIST)", "한밭대학교", "대전대학교", "배재대학교", 
        "우송대학교", "을지대학교(대전캠퍼스)", "침례신학대학교", "한남대학교"
    ],
    "울산": [
        "울산대학교", "울산과학기술원(UNIST)", "울산과학대학교"
    ],
    "세종": [
        "고려대학교(세종캠퍼스)", "홍익대학교(세종캠퍼스)"
    ],
    "강원": [
        "강원대학교", "연세대학교(미래캠퍼스)", "강릉원주대학교", "한림대학교", "춘천교육대학교", 
        "강원도립대학교", "상지대학교", "가톨릭관동대학교", "경동대학교", "한라대학교"
    ],
    "충북": [
        "충북대학교", "청주대학교", "서원대학교", "세명대학교", "충주대학교", "극동대학교", 
        "중원대학교", "건국대학교(글로컬캠퍼스)", "한국교통대학교"
    ],
    "충남": [
        "충남대학교", "공주대학교", "순천향대학교", "남서울대학교", "건양대학교", "백석대학교", 
        "호서대학교", "선문대학교", "한서대학교", "나사렛대학교", "중부대학교", "청운대학교"
    ],
    "전북": [
        "전북대학교", "전주대학교", "원광대학교", "군산대학교", "우석대학교", "예수대학교", 
        "한일장신대학교", "호원대학교"
    ],
    "전남": [
        "전남대학교(여수캠퍼스)", "순천대학교", "목포대학교", "동신대학교", "세한대학교", 
        "초당대학교", "목포해양대학교"
    ],
    "경북": [
        "경북대학교(상주캠퍼스)", "포항공과대학교(POSTECH)", "안동대학교", "경주대학교", 
        "김천대학교", "대구가톨릭대학교(경산캠퍼스)", "동국대학교(경주캠퍼스)", 
        "위덕대학교", "한동대학교"
    ],
    "경남": [
        "경상국립대학교", "창원대학교", "인제대학교", "경남대학교", "영산대학교", "울산대학교", 
        "한국해양대학교(통영캠퍼스)", "진주교육대학교"
    ],
    "제주": [
        "제주대학교", "제주국제대학교", "탐라대학교"
    ]
}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return session_memories[session_id]

def extract_regions(text: str) -> List[str]:
    nouns = kkma.nouns(text)
    extracted_regions = []
    for noun in nouns:
        for region, aliases in region_dict.items():
            if noun in aliases:
                extracted_regions.append(region)
                break
    return extracted_regions

def get_universities_by_region(regions: List[str]) -> Dict[str, List[str]]:
    result = {}
    for region in regions:
        if region in university_by_region:
            result[region] = university_by_region[region]
    return result

def preprocess_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    regions = extract_regions(text)
    for region in regions:
        text = text.replace(region, f"[REGION]{region}[/REGION]")
    
    return text.strip()

def setup_langchain():
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    documents = []
    for i, page in enumerate(pages):
        content = preprocess_text(page.page_content)
        regions = extract_regions(content)
        doc = Document(page_content=content, metadata={"page": i+1, "regions": regions})
        documents.append(doc)

    # FAISS 벡터 검색
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # BM25 키워드 검색
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    # 하이브리드 검색기 설정
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    retriever_tool = create_retriever_tool(
        ensemble_retriever,
        "pdf_search",
        "외국인 특별전형 시행계획 주요사항 PDF 파일에서 추출한 정보를 검색할 때 이 툴을 사용하세요. 지역명이 언급된 경우 해당 지역의 대학교 정보를 함께 제공합니다."
    )

    tools = [retriever_tool]

    openai = ChatOpenAI(
        model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 한국 대학의 외국인 특별전형에 대한 전문가입니다. 주어진 PDF 문서의 내용을 바탕으로 질문에 답변해야 합니다.
        답변 시 다음 지침을 따르세요:
        1. PDF 문서의 정보만을 사용하여 답변하세요.
        2. 문서에 없는 정보에 대해서는 "해당 정보는 제공된 문서에 없습니다"라고 답변하세요.
        3. 지원 자격, 전형 방법, 제출 서류, 전형 일정 등에 대해 구체적으로 답변하세요.
        4. 대학별 특징이나 차이점이 있다면 언급하세요.
        5. 답변은 친절하고 명확하게 제공하세요.
        6. 필요한 경우 답변을 목록화하여 제시하세요.
        7. 이전 대화 내용을 참고하여 일관성 있는 답변을 제공하세요.
        8. 표의 내용을 참조할 때는 행과 열의 관계를 명확히 설명하세요.
        9. 질문에 언급된 지역과 관련된 대학교 정보가 있다면 반드시 포함하여 답변하세요.
        10. 제공된 대학교 목록을 참고하여 관련 대학들의 정보를 포함해 답변하세요.
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행할 코드
    global agent_executor
    agent_executor = setup_langchain()
    yield
    # 종료 시 실행할 코드 (필요한 경우)

app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def query(query: Query):
    try:
        memory = get_memory(query.session_id)
        
        regions = extract_regions(query.input)
        universities = get_universities_by_region(regions)
        
        university_info = "\n".join([f"{region}: {', '.join(unis)}" for region, unis in universities.items()])
        enhanced_query = f"{query.input}\n추출된 지역 정보: {regions}\n관련 대학교:\n{university_info}"
        
        response = agent_executor.invoke(
            {"input": enhanced_query, "chat_history": memory.chat_memory.messages},
            {"memory": memory}
        )
        
        # 대학교 정보를 응답에 추가
        final_response = f"{response['output']}\n\n추가 대학교 정보:\n{university_info}"
        
        memory.chat_memory.add_user_message(query.input)
        memory.chat_memory.add_ai_message(final_response)
        
        return {"response": final_response, "extracted_regions": regions, "related_universities": universities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))