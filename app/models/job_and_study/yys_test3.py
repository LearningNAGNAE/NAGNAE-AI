from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from functools import partial
from pydantic import BaseModel, Field
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
from typing import List, Dict, Callable, Any, Optional
from langchain_community.retrievers.elastic_search_bm25 import ElasticSearchBM25Retriever as ElasticsearchRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import BaseRetriever, Document
from contextlib import asynccontextmanager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elasticsearch import helpers  # bulk 작업을 위해 필요합니다
from elasticsearch import Elasticsearch  # 이 줄을 추가합니다
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import AIMessage
from collections import defaultdict
import requests, json, spacy, fasttext, traceback
from langchain.schema.runnable import RunnableConfig
from datetime import datetime
from elasticsearch import AsyncElasticsearch

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
    # print(f"Error: Model file '{model_path}' not found. Please download it first.")
    exit(1)

try:
    model = fasttext.load_model(model_path)
except ValueError as e:
    # print(f"Error loading model: {e}")
    exit(1)

# Session memories storage
session_memories = {}

# Initialize KoNLPy
kkma = Kkma()


elasticsearch_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    당신은 한국 대학의 재외국민과 외국인 특별전형에 대한 정보를 검색하는 시스템입니다.
    다음 질문에 관련된 정보를 찾아주세요:

    질문: {query}

    검색 시 고려해야 할 사항:
    1. 대학명, 지역명, 전형명 등의 키워드를 중점적으로 찾으세요.
    2. 모집 인원, 지원 자격, 전형 방법 등의 구체적인 정보를 포함하는 문서를 우선적으로 찾으세요.
    3. PDF 문서의 페이지 번호나 섹션 정보도 함께 찾아주세요.
    4. 최신 정보를 우선적으로 찾되, 과거 정보도 참고할 수 있도록 해주세요.
    5. 데이터출처를 명확히 밝혀주세요.

    이 지침을 바탕으로 관련성 높은 검색 결과를 제공해주세요.
    """
)


def fetch_openapi_data(url: str):
    response = requests.get(url)
    print(response.json());
    print("fds패치데이터")
    return response.json()

def fetch_major_seq_numbers(api_key, per_page=30000):
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={api_key}&svcType=api&svcCode=MAJOR&contentType=json&gubun=univ_list&thisPage=1&perPage={per_page}"
    response = requests.get(url)
    data = response.json()
    return [item['majorSeq'] for item in data['dataSearch']['content']]

def fetch_major_details(api_key, major_seq):
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={api_key}&svcType=api&svcCode=MAJOR_VIEW&contentType=json&gubun=univ_list&majorSeq={major_seq}"
    response = requests.get(url)
    return response.json()

async def embed_and_store_openapi_data1(data: dict):
    content_list = data.get('dataSearch', {}).get('content', [])
    
    for item1 in content_list:
        # Convert the item to a JSON string
        text = json.dumps(item1)
        
        # Generate embedding for the text
        vector = embeddings.embed_query(text)
        
        # Create a document with structured data and embedding
        document = {
            "content": item1,  # Store the original structured data
            "content_vector": vector,
            "metadata": {
                "source": "openapi",
                "timestamp": datetime.datetime.now().isoformat(),
                "campusName": item1.get("campusName"),
                "schoolName": item1.get("schoolName"),
                "schoolType": item1.get("schoolType"),
                "region": item1.get("region"),
                "estType": item1.get("estType")
            }
        }
    
    try:
        result = await es_client.index(index="openapi_data1", body=document)
        print(f"Document indexed: {result['result']}")
    except Exception as e:
        print(f"Error indexing document: {str(e)}")

async def embed_and_store_openapi_data2(data: dict):
    content_list = data.get('dataSearch', {}).get('content', [])
    
    for item2 in content_list:
        # Convert the item to a JSON string
        text = json.dumps(item2)
        
        # Generate embedding for the text
        vector = embeddings.embed_query(text)
        
        # Create a document with structured data and embedding
        document = {
            "content": item2,  # Store the original structured data
            "content_vector": vector,
            "metadata": {
                "source": "openapi",
                "timestamp": datetime.datetime.now().isoformat(),
                "lClass": item2.get("lClass"),
                "facilName": item2.get("facilName"),
                "majorSeq": item2.get("majorSeq"),
                "mClass": item2.get("mClass")
            }
        }
    
    try:
        result = await es_client.index(index="openapi_data2", body=document)
        print(f"Document indexed: {result['result']}")
    except Exception as e:
        print(f"Error indexing document: {str(e)}")

async def embed_and_store_openapi_data3(data: dict):
    content_list = data.get('dataSearch', {}).get('content', [])
    
    for item in content_list:
        # Convert the item to a JSON string
        text = json.dumps(item)
        
        # Generate embedding for the text
        vector = embeddings.embed_query(text)
        
        # Create a document with structured data and embedding
        document = {
            "content": text,
            "content_vector": vector,
            "metadata": {
                "source": "openapi",
                "timestamp": datetime.now().isoformat(),
                "major": item.get("major"),
                "salary": item.get("salary"),
                "employment": item.get("employment"),
                "department": item.get("department"),
                "summary": item.get("summary"),
                "relate_subject": item.get("relate_subject"),
                "career_act": item.get("career_act"),
                "job": item.get("job"),
                "qualifications": item.get("qualifications"),
                "interest": item.get("interest"),
                "property": item.get("property"),
                "enter_field": item.get("enter_field"),
                "main_subject": item.get("main_subject"),
                "university": item.get("university"),
                "chartData": item.get("chartData"),
                "GenCD": item.get("GenCD"),
                "SchClass": item.get("SchClass"),
                "lstMiddleAptd": item.get("lstMiddleAptd"),
                "lstHighAptd": item.get("lstHighAptd"),
                "lstVals": item.get("lstVals")
            }
        }
    
        try:
            result = await es_client.index(index="openapi_data3", body=document)
            print(f"Document indexed: {result['result']}")
        except Exception as e:
            print(f"Error indexing document: {str(e)}")

def detect_language_fasttext(text):
    predictions = model.predict(text, k=1)  # 가장 가능성 높은 언어 1개 반환

    # print(f"언어 감지 메소드 : {text}")
    return predictions[0][0].replace('__label__', '')

async def multi_index_search(query: str, es_client, embeddings, k: int = 5) -> List[Dict]:
    es_prompt = elasticsearch_prompt.format(query=query)
    query_vector = embeddings.embed_query(es_prompt)
    
    # 모든 관련 인덱스 포함
    index_names = ["openapi_data1", "openapi_data2", "openapi_data3"]
    
    # 각 인덱스에 대한 가중치 설정
    index_weights = {
        "openapi_data1": 0.4,
        "openapi_data2": 0.3,
        "openapi_data3": 0.3
    }
    
    script_score_query = {
        "script_score": {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query, "boost": 1.0}}},
                        {"match": {"content": {"query": es_prompt, "boost": 0.5}}}
                    ]
                }
            },
            "script": {
                "source": """
                    double score = _score;
                    if (params.index_weights.containsKey(doc['_index'].value)) {
                        score *= params.index_weights[doc['_index'].value];
                    }
                    return score + cosineSimilarity(params.query_vector, 'content_vector') * 10.0;
                """,
                "params": {
                    "query_vector": query_vector,
                    "index_weights": index_weights
                }
            }
        }
    }
    
    existing_indices = [index for index in index_names if await es_client.indices.exists(index=index)]
    
    print(existing_indices, 'fdsfdsfsf');

    if not existing_indices:
        print("No existing indices found");
        return []
    
    try:
        results = await es_client.search(
            index=existing_indices,
            body={"query": script_score_query, "size": k}
        )
        
        return [{"content": hit["_source"]["content"], "score": hit["_score"], "index": hit["_index"]} for hit in results["hits"]["hits"]]
    
    except Exception as e:
        print(f"Error during Elasticsearch search: {str(e)}");
        return []
    
def perform_ner(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 디버깅 정보 출력
    # print(f"NER output: {entities} and text: {text}")
    
    # 엔티티를 레이블별로 정리 (예: 'ORG', 'LOC', 'PERSON' 등)
    organized_entities = defaultdict(list)
    for entity, label in entities:
        organized_entities[label].append(entity)
    
    # 레이블별로 엔티티 출력
    for label, entity_list in organized_entities.items():
        # print(f"Label: {label}, Entities: {', '.join(entity_list)}")
    
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
    
    # print(f"Region Entities: {region_entities}")
    # print(f"University Entities: {university_entities}")
    
    return entities, region_entities, university_entities

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # print(f"session_memories 확인용 : {session_memories} , 세션아이디 확인: {session_id}");
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
        "01": {"title": "대학입학전형기본사항", "start": 1, "end": 10},
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
            doc = Document(page_content=content, metadata={
                "page": page.metadata["page"], 
                "regions": regions,
                "section": section,
                "title": info["title"]
            })
            processed_pages.append(doc)
    
    return processed_pages

async def process_and_save_data():
    global es_client

    # print("Starting data processing and saving...")
    
    # OpenAPI 데이터 처리
    openapi_index = ["openapi_data1", "openapi_data2", "openapi_data3"]


    if not es_client.indices.exists(index=openapi_index):
        print(f"Creating index: {openapi_index}")
        es_client.indices.create(index=openapi_index, body=openapi_mappings)
    else:
        print(f"Index {openapi_index} already exists")

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
    
    api_key = "666d4a31ff13fb263681a0a245cf0cb6"
    print(api_key);
    # 1. 대학 목록 API
    university_url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={api_key}&svcType=api&svcCode=SCHOOL&contentType=json&gubun=univ_list&thisPage=1&perPage=20000"
    university_data = await fetch_openapi_data(university_url)
    print(university_data);
    await embed_and_store_openapi_data1(university_data)
    print(f"University data sample: {json.dumps(university_data[:2], indent=2)}")


    # 2. 전공 목록 API
    major_url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={api_key}&svcType=api&svcCode=MAJOR&contentType=json&gubun=univ_list&thisPage=1&perPage=30000"
    major_data = await fetch_openapi_data(major_url)
    await embed_and_store_openapi_data2(major_data)
    print(f"Major data sample: {json.dumps(major_data[:2], indent=2)}")

    # 3. 전공 상세 정보 API
    major_seq_numbers = await fetch_major_seq_numbers(api_key)
    for major_seq in major_seq_numbers:
        major_details = await fetch_major_details(api_key, major_seq)
        await embed_and_store_openapi_data3(major_details)
        # print(f"Processed and stored data for majorSeq: {major_seq}")

    # 저장 후 인덱스 새로고침
    es_client.indices.refresh(index=openapi_index)

    print("Sample of university data:")
    print(json.dumps(university_data[:2], indent=2))  # 처음 2개 항목만 출력

    # 저장된 문서 수 확인
    count = es_client.count(index=openapi_index)
    print(f"Documents in {openapi_index}: {count['count']}")

    # PDF 처리 (기존 코드)
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

async def initialize_langchain(language='ko'):
    global agent_executor

    class FunctionRetriever(BaseRetriever):
        async def aget_relevant_documents(self, query: str) -> List[Document]:
            results = await multi_index_search(query, es_client, embeddings, k=5)
            return [Document(page_content=result['content'], metadata={'source': result['index']}) for result in results]

        def get_relevant_documents(self, query: str) -> List[Document]:
            return asyncio.run(self.aget_relevant_documents(query))

    retriever = FunctionRetriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "combined_search",
        "이 도구는 PDF 파일과 OpenAPI 데이터에서 대학 입학 전형 기본 사항, 대학별 모집 유형, 모집 단위 및 모집 인원, 전형 방법 및 전형 요소, 대학교명, 대략적인 대학 위치, 전공 정보 등의 구체적인 정보를 검색하는 데 사용됩니다."
    )

    tools = [retriever_tool]

    openai = ChatOpenAI(
        model="gpt-3.5-turbo", 
        api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=0.1,
        max_tokens=1000
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 한국 대학의 재외국민과 외국인 특별전형 전문가입니다..."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def manage_chat_history(memory: ConversationBufferMemory, max_messages: int = 2, max_tokens: int = 200):
    messages = memory.chat_memory.messages
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    
    total_tokens = sum(len(m.content.split()) for m in messages)
    while total_tokens > max_tokens and len(messages) > 2:
        removed = messages.pop(0)
        total_tokens -= len(removed.content.split())
    
    memory.chat_memory.messages = messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, es_client
    print("Lifespan function started")
    elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    
    if not elasticsearch_url.startswith(('http://', 'https://')):
        elasticsearch_url = f"http://{elasticsearch_url}"
    
    try:
        es_client = AsyncElasticsearch([elasticsearch_url])
        
        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")


        if not es_client.indices.exists(index="openapi_data") or not es_client.indices.exists(index="pdf_search"):
            print("Indices do not exist. Running process_and_save_data()")
            await process_and_save_data()
        else:
            print("Indices already exist")

        print("Initializing langchain")
        agent_executor = await initialize_langchain('ko')
        print("Langchain initialized")
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


@app.post("/study")
async def query(query: Query):
    global agent_executor
    if agent_executor is None:
        return {"message": "서버 초기화 중입니다. 잠시 후 다시 시도해 주세요."}
    
    try:
        print(f"Received query: {query.input}")
        search_results = await multi_index_search(query.input, es_client, embeddings, k=5)
        print(f"Search results: {search_results}")
        
        context = "\n".join([result['content'] for result in search_results])
        print(f"Context: {context[:100]}...")  # 처음 100자만 출력
        
        response = await agent_executor.ainvoke({
            "input": f"{query.input}\n\n컨텍스트: {context}",
            "chat_history": []
        })
        print(f"Agent response: {response}")

        if isinstance(response, dict) and "output" in response:
            final_response = response["output"]
        else:
            final_response = str(response)

        return {"response": final_response}
    
    except Exception as e:
        print(f"Error in /study endpoint: {str(e)}")
        print(traceback.format_exc())
        error_message = f"내부 서버 오류: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)