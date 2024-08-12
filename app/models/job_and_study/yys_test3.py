from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from functools import partial
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import re
from konlpy.tag import Kkma
from typing import List, Dict
from langchain_community.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever as ElasticsearchRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import BaseRetriever, Document
from typing import List
from contextlib import asynccontextmanager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elasticsearch import helpers  # bulk 작업을 위해 필요합니다
from elasticsearch import Elasticsearch  # 이 줄을 추가합니다
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import spacy
import json
import fasttext
from langchain.schema import AIMessage
import traceback
from collections import defaultdict
import requests, datetime

load_dotenv()

agent_executor = None
es_client = None
es_retriever = None
embeddings = OpenAIEmbeddings()

# spaCy 모델 로드
nlp = spacy.load("ko_core_news_sm")

app = FastAPI()

class Query(BaseModel):
    input: str
    session_id: str

pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\2025학년도 재외국민과 외국인 특별전형 시행계획 주요사항.pdf"

# FastText 모델
# 현재 스크립트의 디렉토리 경로를 가져옵니다
fasttext_current_dir = os.path.dirname(os.path.abspath(__file__))

# 모델 파일의 경로를 현재 디렉토리를 기준으로 설정합니다
model_path = os.path.join(fasttext_current_dir, 'lid.176.bin')

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please download it first.")
    exit(1)

try:
    model = fasttext.load_model(model_path)
except ValueError as e:
    print(f"Error loading model: {e}")
    exit(1)

# Session memories storage
session_memories = {}

# Initialize KoNLPy
kkma = Kkma()



def fetch_openapi_data(url: str):
    response = requests.get(url)
    return response.json()

def embed_and_store_openapi_data(data: dict):
    text = json.dumps(data)
    vector = embeddings.embed_query(text)
    
    document = {
        "content": text,
        "content_vector": vector,
        "metadata": {
            "source": "openapi",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    es_client.index(index="openapi_data", body=document)

def detect_language_fasttext(text):
    predictions = model.predict(text, k=1)  # 가장 가능성 높은 언어 1개 반환

    print(f"언어 감지 메소드 : {text}")
    return predictions[0][0].replace('__label__', '')

def ollama_search(query: str, es_client, language='ko') -> str:
    ollama = Ollama(model="gemma2:latest")
    
    # 벡터 DB(Elasticsearch) 검색
    query_vector = embeddings.embed_query(query)
    script_score_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    es_response = es_client.search(
        index=["pdf_search", "openapi_data"],
        body={
            "query": script_score_query,
            "size": 5
        }
    )
    
    context = "\n".join([hit["_source"]["content"] for hit in es_response["hits"]["hits"]])
    
    prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=
    f"""
    당신은 한국 대학의 재외국민과 외국인 특별전형 전문가입니다. 제공된 컨텍스트와 데이터를 바탕으로 간결하고 정확하게 답변하세요.
    
    컨텍스트:
    {{context}}
    
    주요 지침:
    - 사용자 질문을 정확히 파악하고 요점만 답변하세요.
    - 대학교 목록이나 모집단위를 요청받으면 모든 대학교와 모집단위를 나열하세요.
    - 숫자 정보(대학 수, 모집 인원 등)를 물으면 정확한 숫자만 답변하세요.
    - 불필요한 설명이나 부가 정보는 제공하지 마세요.
    - 정보가 없으면 "해당 정보를 찾을 수 없습니다."라고만 답변하세요.
    - 중복된 정보는 생략하고 한 번만 제공하세요.
    - {language} 언어로 응답하세요.

    질문: {{query}}
    """
    )
    
    chain = LLMChain(llm=ollama, prompt=prompt)
    response = chain.run(query=query, context=context)
    
    print(f"ollama_search 확인 : {response}")

    return response

def perform_ner(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 디버깅 정보 출력
    print(f"NER output: {entities} and text: {text}")
    
    # 엔티티를 레이블별로 정리 (예: 'ORG', 'LOC', 'PERSON' 등)
    organized_entities = defaultdict(list)
    for entity, label in entities:
        organized_entities[label].append(entity)
    
    # 레이블별로 엔티티 출력
    for label, entity_list in organized_entities.items():
        print(f"Label: {label}, Entities: {', '.join(entity_list)}")
    
    # 지역 엔티티와 대학교명 구분하기
    region_entities = []
    university_entities = []
    
    for entity, _ in entities:
        matched_region = None
        for region, data in region_data.items():
            if any(alias in entity for alias in data['aliases']):
                matched_region = region
                break
        
        if matched_region:
            if entity in region_data[matched_region]['universities']:
                university_entities.append((entity, matched_region))
            else:
                region_entities.append((entity, matched_region))
    
    print(f"Region Entities: {region_entities}")
    print(f"University Entities: {university_entities}")
    
    return entities, region_entities, university_entities

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print(f"session_memories 확인용 : {session_memories} , 세션아이디 확인: {session_id}");
    return session_memories[session_id]

def load_region_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'regions.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

region_data = load_region_data()

def extract_regions(text: str) -> List[str]:
    extracted_regions = []
    for region, data in region_data.items():
        if any(alias in text for alias in data['aliases']):
            extracted_regions.append(region)
    return extracted_regions

def get_universities_by_region(regions: List[str]) -> Dict[str, List[str]]:
    return {region: region_data[region]['universities'] for region in regions if region in region_data}

def preprocess_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    regions = extract_regions(text)
    for region in regions:
        text = text.replace(region, f"[REGION]{region}[/REGION]")
    
    return text.strip()

def process_pdf_pages(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    sections = {
        "01": {"title": "2025학년도 대학입학전형기본사항", "start": 1, "end": 10},
        "02": {"title": "대학별 모집유형", "start": 11, "end": 20},
        "03": {"title": "모집단위 및 모집인원", "start": 21, "end": 166},
        "04": {"title": "전형방법 및 전형요소", "start": 167, "end": len(pages)}
    }
    
    processed_pages = []
    for section, info in sections.items():
        section_pages = pages[info["start"]-1:info["end"]]
        for page in section_pages:
            content = preprocess_text(page.page_content)
            regions = extract_regions(content)
            summary = summarize_text(content)
            doc = Document(page_content=summary, metadata={
                "page": page.metadata["page"], 
                "regions": regions,
                "section": section,
                "title": info["title"]
            })
            processed_pages.append(doc)
    
    return processed_pages

def process_and_save_data():
    global es_client

    # OpenAPI 데이터 처리
    openapi_index = "openapi_data"
    openapi_mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "content_vector": {"type": "dense_vector", "dims": 1536},
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "timestamp": {"type": "date"}
                    }
                }
            }
        }
    }

    if not es_client.indices.exists(index=openapi_index):
        es_client.indices.create(index=openapi_index, body=openapi_mappings)
    
    # OpenAPI 데이터 가져오기 및 저장
    openapi_url = "https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey=666d4a31ff13fb263681a0a245cf0cb6&svcType=api&svcCode=SCHOOL&contentType=json&gubun=univ_list&thisPage=1&perPage=20000"  # 실제 OpenAPI URL로 변경
    openapi_data = fetch_openapi_data(openapi_url)
    embed_and_store_openapi_data(openapi_data)

    es_client.indices.refresh(index=openapi_index)

    index_name = "pdf_search"
    
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "content_vector": {"type": "dense_vector", "dims": 1536},
                "metadata": {
                    "properties": {
                        "page": {"type": "integer"},
                        "regions": {"type": "keyword"},
                        "section": {"type": "keyword"},
                        "title": {"type": "text"}
                    }
                }
            }
        }
    }

    
    
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=mappings)
    
    processed_pages = process_pdf_pages(pdf_path)
    
    actions = []
    for doc in processed_pages:
        embedding = embeddings.embed_query(doc.page_content)
        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_source": {
                "content": doc.page_content,
                "content_vector": embedding,
                "metadata": doc.metadata
            }
        })
    
    if actions:
        helpers.bulk(es_client, actions)
    
    es_client.indices.refresh(index=index_name)

def hybrid_search(query: str, top_k: int = 6, es_weight: float = 0.3, ollama_weight: float = 0.2, openapi_weight: float = 0.5):
    query_vector = embeddings.embed_query(query)
    
    script_score_query = {
        "script_score": {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": query}}
                    ]
                }
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    try:
        es_response = es_client.search(
            index="pdf_search",
            body={
                "query": script_score_query,
                "size": top_k
            }
        )
        
        openapi_response = es_client.search(
            index="openapi_data",
            body={
                "query": script_score_query,
                "size": top_k
            }
        )
        
        ollama_result = ollama_search(query, es_client)
        
        es_results = [{"source": "elasticsearch", "content": hit["_source"]["content"], "score": hit["_score"] * es_weight, "metadata": hit["_source"]["metadata"]} for hit in es_response["hits"]["hits"]]
        openapi_results = [{"source": "openapi", "content": hit["_source"]["content"], "score": hit["_score"] * openapi_weight, "metadata": hit["_source"]["metadata"]} for hit in openapi_response["hits"]["hits"]]
        ollama_results = [{"source": "ollama", "content": ollama_result, "score": ollama_weight}]
        
        combined_results = es_results + openapi_results + ollama_results
        combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return combined_results
    except Exception as e:
        print(f"검색 중 오류 발생: {str(e)}")
        return []
    
def initialize_langchain(language='ko'):
    global agent_executor, es_client, es_retriever
    
    es_retriever = ElasticsearchRetriever(
        client=es_client,
        index_name="pdf_search",
        k=5
    )
    
    retriever_tool = create_retriever_tool(
        es_retriever,
        "pdf_search",
        "이 도구는 PDF 파일에서 대학 입학 전형 기본 사항, 대학별 모집 유형, 모집 단위 및 모집 인원, 전형 방법 및 전형 요소, 대학교명, 대략적인 대학 위치 등의 구체적인 정보를 검색하는 데 사용됩니다."
    )

    tools = [retriever_tool]

    openai = ChatOpenAI(
        model="gpt-3.5-turbo", 
        api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=0.1,
        max_tokens=1000
    )

    prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    f"""
    당신은 한국 대학의 재외국민과 외국인 특별전형 전문가입니다. 제공된 PDF 문서, university_by_region 데이터, region_dict 데이터를 바탕으로 간결하고 정확하게 답변하세요.
    주요 지침:

    - 사용자 질문을 정확히 파악하고 요점만 답변하세요.
    - 질문자가 매번 질문을 할 때마다 university_by_region과 region_dict를 반드시 확인하고 답변하세요.
    - 대학교 목록이나 모집단위를 요청받으면 모든 대학교와 모집단위를 나열하세요. 절대로 일부만 나열하고 "등"으로 끝내지 마세요.
    - 숫자 정보(대학 수, 모집 인원 등)를 물으면 반드시 정확한 숫자를 제공하세요.
    - 특정 대학의 전공이나 모집 인원 정보는 PDF 문서의 21페이지부터 166페이지를 참조하여 정확히 제공하세요.
    - 대학별 모집유형 정보는 PDF의 11페이지에서 20페이지를 참고하세요.
    - 대학입학전형기본사항은 PDF의 1페이지부터 10페이지를 참고하세요.
    - 불필요한 설명이나 부가 정보는 제공하지 마세요.
    - 정보가 없으면 "해당 정보를 찾을 수 없습니다."라고만 답변하세요.
    - 외국인 전형 정보를 물어보면 반드시 PDF의 '전형방법 및 전형요소' 섹션(167페이지 이후)을 참고하여 답변하세요.
    - 중복된 정보는 생략하고 한 번만 제공하세요.
    - 요청받은 정보에 대해 가능한 한 완전하고 포괄적인 답변을 제공하세요. 정보를 생략하지 말고 알고 있는 모든 관련 정보를 포함해 주세요.
    - 답변 시 정확한 정보만을 제공하세요. 불확실한 정보는 제공하지 마세요.
    - {language} 언어로 응답하세요.

    답변 예시:
    User: 충북에 외국인전형이 있는 대학들을 알려줘
    Assistant: 충북에 외국인전형이 있는 대학들은 다음과 같습니다:
    1. 충북대학교
    2. 청주대학교
    3. 서원대학교
    4. 세명대학교
    5. 청주교육대학교
    6. 한국교통대학교
    7. 극동대학교
    8. 중원대학교
    9. 건국대학교 글로컬캠퍼스

    총 9개의 대학이 충북에서 외국인전형을 실시하고 있습니다.

    User: 경북에 있는 대학교 수는?
    Assistant: 9개입니다.
    User: 위덕대학교에 외국인 전형이 있는 전공을 알려줘
    Assistant: 불교문화학과, 한국어학부, 일본언어문화학과, 경찰정보보안학과, 경영학과, 사회복지학과, 항공호텔서비스학과, 유아교육과, 외식조리제과제빵학부, 지능형전력시스템공학과, 건강스포츠학부입니다.
    User: 군산대학교(전북)의 외국인 전형 전공들의 모집 인원을 알려줘
    Assistant: 33명입니다.
    User: 서울대학교의 외국인 전형 방법을 알려줘
    Assistant: 서류평가 100%입니다.
    User: 서강대학교의 외국인 전형이 있는지 알려줘
    Assistant: 있습니다. 모집단위로는 국제인문학부, 사회과학부, 경제학부, 경영학부, 자연과학부, 공학부, 컴퓨터공학과, 전자공학과, 국어국문학과, 영미어문, 유럽문화, 중국문화 등이 있습니다.
    User: 경남에 있는 대학교들을 알려줘
    Assistant: 경상국립대학교, 창원대학교, 경남대학교, 인제대학교입니다.
    User: 경상국립대학교의 외국인 전형 전공을 알려줘
    Assistant: 국어국문학과, 영어영문학과, 독일학과, 러시아학과, 중어중문학과, 사학과, 철학과, 불어불문학과, 일어일문학과, 민속무용학과, 한문학과, 법학과, 행정학과, 정치외교학과, 사회학과, 경제학과, 경영학부, 회계학과, 국제통상학과, 심리학과, 사회복지학과, 아동가족학과, 시각디자인학과 등이 있습니다.
    """
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    extract_prompt = PromptTemplate(
        input_variables=["query"],
        template="다음 질문에서 중요한 키워드와 의도를 추출하세요: {query}"
    )
    print(f"extract_prompt 확인용: {extract_prompt}");

    response_prompt = PromptTemplate(
        input_variables=["extracted_info", "agent_response"],
        template="""
        당신은 한국 대학의 재외국민과 외국인 특별전형 전문가입니다. 제공된 PDF 문서, university_by_region 데이터, region_dict 데이터를 바탕으로 간결하고 정확하게 답변하세요.
        주요 지침:

        - 사용자 질문을 정확히 파악하고 요점만 답변하세요.
        - 질문자가 매번 질문을 할 때마다 university_by_region과 region_dict를 반드시 확인하고 답변하세요.
        - 대학교 목록이나 모집단위를 요청받으면 모든 대학교와 모집단위를 나열하세요. 단, 모집단위가 너무 많으면 일부만 보여주고 "등"을 붙이세요. 대학교명을 요청받으면 모든 대학교명을 생략하지 말고 다 알려주세요.
        - 숫자 정보(대학 수, 모집 인원 등)를 물으면 정확한 숫자만 답변하세요.
        - 특정 대학의 전공이나 모집 인원 정보는 PDF 문서의 21페이지부터 166페이지를 참조하여 정확히 제공하세요.
        - 대학별 모집유형 정보는 PDF의 11페이지에서 20페이지를 참고하세요.
        - 대학입학전형기본사항은 PDF의 1페이지부터 10페이지를 참고하세요.
        - 불필요한 설명이나 부가 정보는 제공하지 마세요.
        - 정보가 없으면 "해당 정보를 찾을 수 없습니다."라고만 답변하세요.
        - 외국인 전형 정보를 물어보면 반드시 PDF의 '전형방법 및 전형요소' 섹션(167페이지 이후)을 참고하여 답변하세요.
        - 중복된 정보는 생략하고 한 번만 제공하세요.

        답변 예시:
        User: 경북에 있는 대학교들을 알려줘
        Assistant: 경북대학교(상주캠퍼스), 포항공과대학교(POSTECH), 안동대학교, 경주대학교, 김천대학교, 대구가톨릭대학교(경산캠퍼스), 동국대학교(경주캠퍼스), 위덕대학교, 한동대학교입니다.
        User: 경북에 있는 대학교 수는?
        Assistant: 9개입니다.
        User: 위덕대학교에 외국인 전형이 있는 전공을 알려줘
        Assistant: 불교문화학과, 한국어학부, 일본언어문화학과, 경찰정보보안학과, 경영학과, 사회복지학과, 항공호텔서비스학과, 유아교육과, 외식조리제과제빵학부, 지능형전력시스템공학과, 건강스포츠학부입니다.
        User: 군산대학교(전북)의 외국인 전형 전공들의 모집 인원을 알려줘
        Assistant: 33명입니다.
        User: 서울대학교의 외국인 전형 방법을 알려줘
        Assistant: 서류평가 100%입니다.
        User: 서강대학교의 외국인 전형이 있는지 알려줘
        Assistant: 있습니다. 모집단위로는 국제인문학부, 사회과학부, 경제학부, 경영학부, 자연과학부, 공학부, 컴퓨터공학과, 전자공학과, 국어국문학과, 영미어문, 유럽문화, 중국문화 등이 있습니다.
        User: 경남에 있는 대학교들을 알려줘
        Assistant: 경상국립대학교, 창원대학교, 경남대학교, 인제대학교입니다.
        User: 경상국립대학교의 외국인 전형 전공을 알려줘
        Assistant: 국어국문학과, 영어영문학과, 독일학과, 러시아학과, 중어중문학과, 사학과, 철학과, 불어불문학과, 일어일문학과, 민속무용학과, 한문학과, 법학과, 행정학과, 정치외교학과, 사회학과, 경제학과, 경영학부, 회계학과, 국제통상학과, 심리학과, 사회복지학과, 아동가족학과, 시각디자인학과 등이 있습니다.

        추출된 정보: {extracted_info}
        에이전트 응답: {agent_response}

        위 정보를 바탕으로 최종 응답을 생성하세요. 생략하지 말고 모든 정보를 포함하여 제공하세요.
        """,
    )
    print(f"response_prompt 확인용: {response_prompt}");

    extract_chain = LLMChain(llm=openai, prompt=extract_prompt, output_key="extracted_info")
    response_chain = LLMChain(llm=openai, prompt=response_prompt, output_key="final_response")

    overall_chain = SequentialChain(
        chains=[extract_chain, response_chain],
        input_variables=["query", "agent_response"],
        output_variables=["extracted_info", "final_response"],
        verbose=True
    )

    print(f"1. {agent_executor}, 2. {overall_chain}");
    
    return agent_executor, overall_chain

def summarize_text(text: str, max_tokens: int = 500) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"주어진 텍스트를 {max_tokens}단어 이내로 간결하게 요약하십시오. 주요 포인트만 포함하도록 합니다."),
        ("human", "{input}")
    ])
    
    openai = ChatOpenAI(
        model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1,
        max_tokens=max_tokens
    )
    
    response = openai(prompt.format_prompt(input=text).to_messages())
    print(f" 요약 정보 : {response.content}");

    return response.content

def manage_chat_history(memory: ConversationBufferMemory, max_messages: int = 2, max_tokens: int = 200):
    messages = memory.chat_memory.messages
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    
    total_tokens = sum(len(m.content.split()) for m in messages)
    while total_tokens > max_tokens and len(messages) > 2:
        removed = messages.pop(0)
        total_tokens -= len(removed.content.split())
    
    memory.chat_memory.messages = messages

def generate_response(query: str, search_results: List[Dict]) -> str:
    context = "\n".join([f"Section {result['metadata']['section']} ({result['metadata']['title']}): {result['content']}" for result in search_results if 'metadata' in result])
    prompt = f"""
    질문: {query}
    
    다음 정보를 참조하여 질문에 답하세요:
    {context}
    
    답변 시 주의사항:
    - 요청받은 정보에 대해 가능한 한 완전하고 포괄적인 답변을 제공하세요.
    - 정보를 생략하지 말고 알고 있는 모든 관련 정보를 포함해 주세요.
    - 대학 목록이나 정보를 요청받았을 때는 해당되는 모든 항목을 나열하세요. 
    - 숫자 정보(예: 대학 수)를 요청받았을 때는 정확한 숫자를 제공하세요.
    - 불확실한 정보는 제공하지 마세요.
    - 추가 정보가 있다면 '추가 정보:'라는 제목 하에 제공해 주세요.
    
    답변:
    """
    
    llm = ChatOpenAI(temperature=0.1)
    response = llm.predict(prompt)
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, es_client
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    
    if not elasticsearch_url.startswith(('http://', 'https://')):
        elasticsearch_url = f"http://{elasticsearch_url}"
    
    print(f"Connecting to Elasticsearch at: {elasticsearch_url}")
    
    try:
        es_client = Elasticsearch([elasticsearch_url])
        
        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
        
        print("Successfully connected to Elasticsearch")

        if not es_client.indices.exists(index="pdf_search"):
            process_and_save_data()  # 여기서 OpenAPI 데이터도 처리됩니다

        
        # 여기서 기본 언어를 'ko'(한국어)로 지정합니다.
        initialize_langchain('ko')
        yield
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print(traceback.format_exc())
        raise
    finally:
        if es_client:
            es_client.close()
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_final_response(query: str, hybrid_results: List[Dict], agent_executor, chat_history, enhanced_query: str) -> str:
    context = "\n".join([f"Source: {result['source']}, Content: {result['content']}" for result in hybrid_results])
    
    summarized_result = summarize_text(context, max_tokens=100)
    
    agent_input = f"{enhanced_query}\n요약: {summarized_result}\n"
    
    agent_response = agent_executor.invoke({
        "input": agent_input, 
        "chat_history": chat_history
    })
    
    if isinstance(agent_response, dict) and "output" in agent_response:
        final_response = agent_response["output"]
    elif isinstance(agent_response, AIMessage):
        final_response = agent_response.content
    else:
        final_response = str(agent_response)
    
    return final_response
@app.post("/study")
def query(query: Query, background_tasks: BackgroundTasks):
    global agent_executor
    if agent_executor is None:
        return {"message": "서버 초기화 중입니다. 잠시 후 다시 시도해 주세요."}
    
    try:
        language = detect_language_fasttext(query.input)
        memory = get_memory(query.session_id)
        print(f"쿼리 메소드의 언어 감지 : {language}")
        entities = perform_ner(query.input)
        print(f"쿼리 메소드의 엔티티 : {entities}")
        regions = extract_regions(query.input)
        print(f"쿼리 메소드의 지역 : {regions}")
        universities = get_universities_by_region(regions)
        print(f"쿼리 메소드의 대학교 : {universities}")
        
        university_info = "\n".join([f"{region}: {', '.join([uni[0] for uni in unis[:2]])}" for region, unis in universities.items()])
        short_query = query.input[:200]
        enhanced_query = f"{short_query}\n지역: {', '.join(regions)}\n대학교: {', '.join([uni[0] for unis in universities.values() for uni in unis[:1]])}\n인식된 개체: {entities}"
        
        manage_chat_history(memory, max_messages=2, max_tokens=300)
        
        search_results = hybrid_search(query.input, top_k=10)
        
        # 에이전트 실행 전에 agent_executor 초기화
        agent_executor, overall_chain = initialize_langchain(language)
        
        final_response = generate_final_response(query.input, search_results, agent_executor, overall_chain, memory.chat_memory.messages, enhanced_query)
        
        # 응답 길이 제한
        final_response = final_response[:500]
        
        final_response_with_info = f"{final_response}\n\n추가 대학교 정보:\n{university_info[:200]}"
        
        memory.chat_memory.add_user_message(query.input[:100])
        memory.chat_memory.add_ai_message(final_response_with_info[:200])
        
        print(f"Hybrid search results: {search_results}")
        print(f"Final response: {final_response}")

        return {
            "response": final_response_with_info, 
            "extracted_regions": regions, 
            "related_universities": universities,
            "entities": entities,
            "detected_language": language
        }
    except Exception as e:
        print(f"쿼리 함수에서 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")