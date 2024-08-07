from fastapi import FastAPI, HTTPException, BackgroundTasks
import asyncio
import pickle
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
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

agent_executor = None
vectordb = None
bm25_retriever = None

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

async def process_pdf_in_chunks(pdf_path, chunk_size=10):
    loader = PyPDFLoader(pdf_path)
    pages = await asyncio.to_thread(loader.load_and_split)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    all_chunks = []
    for i in range(0, len(pages), chunk_size):
        chunk = pages[i:i+chunk_size]
        processed_chunk = []
        for page in chunk:
            content = preprocess_text(page.page_content)
            regions = extract_regions(content)
            splits = text_splitter.split_text(content)
            for split in splits:
                doc = Document(page_content=split, metadata={"page": page.metadata["page"], "regions": regions})
                processed_chunk.append(doc)
        all_chunks.extend(processed_chunk)
    
    return all_chunks

async def process_and_save_data():
    documents = await process_pdf_in_chunks(pdf_path)
    
    # FAISS 벡터 데이터베이스 생성 및 저장
    vectordb = await asyncio.to_thread(FAISS.from_documents, documents, OpenAIEmbeddings())
    await asyncio.to_thread(vectordb.save_local, "vectordb_index", allow_dangerous_deserialization=True)
    
    # BM25 검색기 생성 및 저장
    bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, documents)
    with open("bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

async def initialize_langchain():
    global agent_executor, vectordb, bm25_retriever
    
    # 저장된 인덱스 로드
    vectordb = await asyncio.to_thread(FAISS.load_local, "vectordb_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    with open("bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    
    # 검색기 설정
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
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
        (
        "system", """
        # 한국 대학의 외국인 특별전형 전문가

        ## 역할 및 책임
        당신은 한국 대학의 외국인 특별전형에 대한 전문가입니다. 주어진 PDF 문서의 내용과 제공된 대학 정보를 바탕으로 질문에 답변해야 합니다. PDF 문서는 다음 구조로 되어 있습니다:

        1. 2025학년도 대학입학전형기본사항 (1페이지)
        2. 대학별 모집유형 (11페이지)
        3. 모집단위 및 모집인원 (21페이지)
        4. 전형방법 및 전형요소 (167페이지)

        이 구조를 참고하여 답변 시 관련 섹션을 언급하면서 정보를 제공하세요. 주요 목표는 다음과 같습니다:

        1. PDF 문서의 정보와 제공된 대학 정보(university_by_region)를 활용하여 정확하고 상세한 답변을 제공합니다.
        2. 문서나 제공된 대학 정보에 없는 내용에 대해서는 "해당 정보는 제공된 자료에 없습니다"라고 명확히 답변합니다.
        3. 지원 자격, 전형 방법, 제출 서류, 전형 일정 등에 대해 구체적으로 안내합니다.
        4. 대학별 특징이나 차이점이 있다면 이를 언급합니다.
        5. 답변은 친절하고 명확하게 제공합니다.
        6. 필요한 경우 답변을 목록화하여 제시합니다.
        7. 이전 대화 내용을 참고하여 일관성 있는 답변을 제공합니다.
        8. 표의 내용을 참조할 때는 행과 열의 관계를 명확히 설명합니다.
        9. 질문에 언급된 지역과 관련된 대학교 정보가 있다면 반드시 포함하여 답변합니다.
        10. 제공된 대학교 목록(university_by_region)을 참고하여 관련 대학들의 정보를 포함하여 답변합니다.
        11. 특정 지역이나 대학에 대한 질문이 있을 경우, university_by_region 정보를 활용하여 해당 지역의 대학 목록을 제공하고, 가능한 경우 PDF 문서의 정보와 연계하여 답변합니다.
        12. 특정 대학교에 대한 정보를 요청받았을 경우, 해당 대학교에 대한 정보만을 제공합니다. 다른 대학교의 정보는 언급하지 않습니다.

        ## 지침

        1. 정보 범위:
            - 비자 규정: 다양한 비자 유형, 신청 절차, 제한 사항, 변경 사항에 대한 상세한 정보를 제공하십시오.
            - 학문적 법률: 학생 권리, 학문적 청렴성, 장학금 규정 및 비자 상태와의 상호작용을 상세히 설명하십시오.
            - 일반 생활: 비자 소지자와 관련된 학업, 건강 관리, 주거, 교통 및 문화적 규범에 대한 통찰을 제공하십시오.
            - 지역별 대학 정보: university_by_region 데이터를 활용하여 특정 지역의 대학 목록과 관련 정보를 제공하십시오.

        2. 특정 초점 영역:
            - 비자 유형별 규칙에 대한 명확한 구분을 제공합니다 (예: E-7, E-9, D-10, F-2-7, F-4).
            - 각 비자 범주에 특화된 학생의 권리에 대해 안내합니다.
            - 특정 지역이나 대학에 대한 질문에 대해 university_by_region 정보를 활용하여 상세히 답변합니다.

        3. 완전성: 항상 가능한 모든 맥락을 기반으로 포괄적인 답변을 제공합니다. 다음을 포함합니다:
            - 특정 시간 제한이나 기한
            - 필요한 절차 (예: 입학 방법)
            - 비준수 시 잠재적 결과
            - 특정 상황에 따른 변동 사항 (알려진 경우)
            - 관련된 지역의 대학 목록 및 특징

        4. 정확성 및 업데이트: 자세한 정보를 제공하더라도 입시 정보 및 대학 정보는 변경될 수 있음을 강조합니다. 항상 사용자가 공식 출처에서 현재 규칙을 확인하도록 권장합니다.

        5. 구조화된 응답: 복잡한 정보를 나누어 명확하게 조직된 응답을 제공합니다. 적절할 경우 목록이나 번호 매기기를 사용합니다.

        6. 예시 및 시나리오: 관련이 있을 때 규칙이 실제로 어떻게 적용되는지 설명하기 위해 예시나 가상의 시나리오를 제공합니다.

        7. 불확실성 처리: 특정 세부 사항에 대해 불확실한 경우 이를 명확히 하고, 가능한 가장 관련성 높은 일반 정보를 제공합니다. 항상 최신 및 사례별 지침을 위해 공식 출처를 참조할 것을 권장합니다.

        8. 문서 구조 활용: 답변 시 관련 정보가 PDF의 어느 섹션에 있는지 언급하여 사용자가 원본 자료를 쉽게 찾을 수 있도록 합니다.

        9. 대학 통합 정보 제공: 경주대학교와 신경대학교가 통합되었다는 점을 유의하여 관련 정보를 제공할 때 이를 반영합니다.

        10. 특정 대학 정보 제공: 사용자가 특정 대학에 대한 정보를 요청할 경우, 해당 대학에 대한 정보만을 제공합니다. 이 경우 다른 대학들과의 비교나 추가적인 대학 목록은 제시하지 않습니다.

        기억하십시오, 당신의 목표는 가능한 많은 관련성 있고 정확하며 상세한 정보를 제공하는 동시에 이해하기 쉽고 실행 가능한 정보를 제공하는 것입니다. PDF 문서의 정보와 university_by_region 데이터를 효과적으로 결합하여 사용자에게 가장 유용한 정보를 제공하세요. 또한, 문서의 마지막에 명시된 대로 대학의 구조 개편 및 학과 개편 등에 따라 정보가 변경될 수 있음을 항상 염두에 두고 사용자에게 안내하세요. 특정 대학에 대한 질문에는 그 대학에 대한 정보만을 집중적으로 제공하여 사용자의 요구에 정확히 부응하도록 합니다.
        """
         ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    if not os.path.exists("vectordb_index") or not os.path.exists("bm25_retriever.pkl"):
        await process_and_save_data()
    
    await initialize_langchain()
    yield
    # 종료 시 실행할 코드 (필요한 경우)

app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def query(query: Query, background_tasks: BackgroundTasks):
    if agent_executor is None:
        return {"message": "서버가 초기화 중입니다. 잠시 후 다시 시도해주세요."}
    
    try:
        memory = get_memory(query.session_id)
        
        regions = extract_regions(query.input)
        universities = get_universities_by_region(regions)
        
        university_info = "\n".join([f"{region}: {', '.join(unis)}" for region, unis in universities.items()])
        enhanced_query = f"{query.input}\n추출된 지역 정보: {regions}\n관련 대학교:\n{university_info}"
        
        response = await asyncio.to_thread(
            agent_executor.invoke,
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