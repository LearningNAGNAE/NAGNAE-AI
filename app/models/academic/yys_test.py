import requests
import os, json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time
from tqdm import tqdm
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever, Document
from typing import List, Union, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import CrossEncoder

load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    input: str
    session_id: str

class Response(BaseModel):
    answer: str


# 엔티티를 정의하는 Pydantic 모델
class Entities(BaseModel):
    universities: List[str] = Field(description="List of university names")
    majors: List[str] = Field(description="List of major names")
    keywords: List[str] = Field(description="List of other relevant keywords")

# 환경 변수 로드
university_api_key = os.getenv("UNIVERSITY_API_KEY")
gpt_api_key = os.getenv("OPENAI_API_KEY")
elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\2025학년도 재외국민과 외국인 특별전형 시행계획 주요사항.pdf"

# CrossEncoder 모델 초기화
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')

# 일관된 값을 위하여 Temperature 0.1로 설정 model은 gpt-4o로 설정
openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=gpt_api_key, temperature=0)

# Elasticsearch 클라이언트 설정
es_client = Elasticsearch([elasticsearch_url])
embedding = OpenAIEmbeddings()

# 세션별 대화 기록을 저장할 딕셔너리
session_histories: Dict[str, List] = {}

# 배치 크기와 요청 간 대기 시간 설정
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 1  # 초 단위

# API의 대학 정보 데이터 수집
def fetch_university_data(per_page=30000):
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={university_api_key}&svcType=api&svcCode=SCHOOL&contentType=json&gubun=univ_list&thisPage=1&perPage={per_page}"
    response = requests.get(url)
    return response.json()

# API의 대학 전공 데이터 수집
def fetch_university_major(per_page=30000):
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={university_api_key}&svcType=api&svcCode=MAJOR&contentType=json&gubun=univ_list&thisPage=1&perPage={per_page}"
    response = requests.get(url)
    return response.json()  # 전체 응답을 반환

# API의 전공 상세 정보 데이터 수집
def fetch_major_details(major_seq):
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={university_api_key}&svcType=api&svcCode=MAJOR_VIEW&contentType=json&gubun=univ_list&majorSeq={major_seq}"
    response = requests.get(url)
    return response.json()

# PDF 파일의 데이터 수집
def load_pdf_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# 언어 감지 함수 정의
def detect_language(text: str) -> str:
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = openai.invoke(messages)
    detected_language = response.content.strip().lower()
    # logger.info(f"Detected language: {detected_language}")
    return detected_language

# 언어 감지 함수 정의2
def korean_language(text: str) -> str:
    system_prompt = "You are a translation expert. Your task is to detect the language of a given text and translate it into Korean. Please provide only the translated text in Korean, without any additional explanations or information."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = openai.invoke(messages)
    koreans_language = response.content.strip().lower()
    # logger.info(f"Detected language: {detected_language}")
    return koreans_language

# 엔티티 추출을 위한 함수
def extract_entities(query: str) -> Entities:

    try:
        parser = PydanticOutputParser(pydantic_object=Entities)
        
        prompt = f"""
        Extract relevant entities from the following query. The entities should include university names, major names, and other relevant keywords.

        Query: {query}

        {parser.get_format_instructions()}
        """

        messages = [
            {"role": "system", "content": "You are an expert in extracting relevant entities from text."},
            {"role": "user", "content": prompt}
        ]

        response = openai.invoke(messages)
        return parser.parse(response.content)
    except Exception as e:
        print(f"엔티티 추출 중 오류 발생: {e}")
        return Entities(universities=[], majors=[], keywords=[])

# 쿼리 생성 함수
def generate_elasticsearch_query(entities: Entities):
    should_clauses = []

    if entities.universities:
        should_clauses.append({"terms": {"metadata.schoolName.keyword": entities.universities}})
    
    if entities.majors:
        should_clauses.append({"terms": {"metadata.major.keyword": entities.majors}})
    
    if entities.keywords:
        should_clauses.append({"match": {"text": " ".join(entities.keywords)}})

    return {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
    }

# 임베딩 데이터를 나눠서 인데스저장
def process_in_batches(data, process_func):
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i+BATCH_SIZE]
        process_func(batch)
        time.sleep(RATE_LIMIT_DELAY)

# 인덱스가 이미 있는지 확인하는 함수
def index_exists(index_name):
    return es_client.indices.exists(index=index_name)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_university_data():
    if index_exists('university_data'):
        print("University data index already exists. Skipping...")
        return
    data = fetch_university_data()
    def process_batch(batch):
        for item in batch:
            text = f"{item['schoolName']} {item.get('campusName', '')} {item.get('schoolType', '')} {item.get('schoolGubun', '')}"
            vector = embedding.embed_query(text)
            metadata = {
                'source': 'university_data',
                'schoolName': item['schoolName'],
            }
            
            # 선택적 필드들을 메타데이터에 추가
            optional_fields = ['campusName', 'collegeInfoUrl', 'schoolType', 'link', 'schoolGubun', 
                               'adres', 'region', 'totalCount', 'estType', 'seq']
            for field in optional_fields:
                if field in item:
                    metadata[field] = item[field]
            
            doc = {
                'text': text,
                'vector': vector,
                'metadata': metadata
            }
            es_client.index(index='university_data', body=doc)

    process_in_batches(data['dataSearch']['content'], process_batch)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_university_major():
    if index_exists('university_major'):
        print("University major index already exists. Skipping...")
        return
    major_data = fetch_university_major()
    if 'dataSearch' not in major_data or 'content' not in major_data['dataSearch']:
        print("Unexpected data structure in university major data")
        return
    
    def process_batch(batch):
        for item in batch:
            text = f"{item.get('lClass', '')} {item.get('facilName', '')} {item.get('mClass', '')}"
            vector = embedding.embed_query(text)
            metadata = {
                'source': 'university_major'
            }
            
            # 선택적 필드들을 메타데이터에 추가
            optional_fields = ['lClass', 'facilName', 'majorSeq', 'mClass', 'totalCount']
            for field in optional_fields:
                if field in item:
                    metadata[field] = item[field]
            
            doc = {
                'text': text,
                'vector': vector,
                'metadata': metadata
            }
            es_client.index(index='university_major', body=doc)

    process_in_batches(major_data['dataSearch']['content'], process_batch)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_major_details():
    if index_exists('major_details'):
        print("Major details index already exists. Skipping...")
        return
    major_data = fetch_university_major()
    if 'dataSearch' not in major_data or 'content' not in major_data['dataSearch']:
        print("Unexpected data structure in university major data")
        return
    
    major_seqs = [item['majorSeq'] for item in major_data['dataSearch']['content']]
    
    def process_batch(batch):
        for major_seq in batch:
            major_data = fetch_major_details(major_seq)
            if 'dataSearch' in major_data and 'content' in major_data['dataSearch'] and major_data['dataSearch']['content']:
                item = major_data['dataSearch']['content'][0]  # Assuming one item per major_seq
                text = f"{item.get('major', '')} {item.get('summary', '')}"
                vector = embedding.embed_query(text)
                metadata = {
                    'source': 'major_details'
                }
                
                # 선택적 필드들을 메타데이터에 추가
                optional_fields = ['major', 'salary', 'employment', 'department', 'summary', 
                                   'job', 'qualifications', 'interest', 'property']
                for field in optional_fields:
                    if field in item:
                        metadata[field] = item[field]
                
                # 리스트 형태의 데이터는 문자열로 변환하여 저장
                list_fields = ['relate_subject', 'career_act', 'enter_field', 'main_subject', 'chartData']
                for field in list_fields:
                    if field in item:
                        metadata[field] = json.dumps(item[field])
                
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': metadata
                }
                es_client.index(index='major_details', body=doc)
            else:
                print(f"No data found for major_seq: {major_seq}")

    process_in_batches(major_seqs, process_batch)

# 처음 임베딩 및 인덱스 생성
def embed_and_index_pdf_data():
    if index_exists('pdf_data'):
        print("PDF data index already exists. Skipping...")
        return
    documents = load_pdf_document(pdf_path)
    def process_batch(batch):
        for doc in batch:
            vector = embedding.embed_query(doc.page_content)
            es_doc = {
                'text': doc.page_content,
                'vector': vector,
                'metadata': {
                    'source': 'pdf',
                    'page': doc.metadata['page']
                }
            }
            es_client.index(index='pdf_data', body=es_doc)

    process_in_batches(documents, process_batch)

# 인덱스 업데이트에 필요한 함수들
def update_indices():
    print("Updating indices...")

    # 대학 정보 업데이트
    update_university_data()

    # 대학 전공 정보 업데이트
    update_university_major()

    # 전공 상세 정보 업데이트
    update_major_details()

    # PDF 데이터 업데이트 (필요한 경우)
    update_pdf_data()

    print("Indices update completed.")

# 업데이트(임베딩 및 인덱스)
def update_university_data():
    new_data = fetch_university_data()
    for item in new_data['dataSearch']['content']:
        query = {
            "query": {
                "match": {
                    "metadata.seq": item['seq']
                }
            }
        }
        result = es_client.search(index="university_data", body=query)

        if result['hits']['total']['value'] == 0:
            # 새로운 데이터 추가
            text = f"{item['schoolName']} {item.get('campusName', '')} {item.get('schoolType', '')} {item.get('schoolGubun', '')}"
            vector = embedding.embed_query(text)
            doc = {
                'text': text,
                'vector': vector,
                'metadata': item
            }
            es_client.index(index='university_data', body=doc)
        else:
            # 기존 데이터 업데이트
            existing_doc = result['hits']['hits'][0]
            if existing_doc['_source']['metadata'] != item:
                text = f"{item['schoolName']} {item.get('campusName', '')} {item.get('schoolType', '')} {item.get('schoolGubun', '')}"
                vector = embedding.embed_query(text)
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': item
                }
                es_client.update(index='university_data', id=existing_doc['_id'], body={'doc': doc})

# 업데이트(임베딩 및 인덱스)
def update_university_major():
    new_data = fetch_university_major()
    for item in new_data['dataSearch']['content']:
        query = {
            "query": {
                "match": {
                    "metadata.majorSeq": item['majorSeq']
                }
            }
        }
        result = es_client.search(index="university_major", body=query)

        if result['hits']['total']['value'] == 0:
            # 새로운 데이터 추가
            text = f"{item.get('lClass', '')} {item.get('facilName', '')} {item.get('mClass', '')}"
            vector = embedding.embed_query(text)
            doc = {
                'text': text,
                'vector': vector,
                'metadata': item
            }
            es_client.index(index='university_major', body=doc)
        else:
            # 기존 데이터 업데이트
            existing_doc = result['hits']['hits'][0]
            if existing_doc['_source']['metadata'] != item:
                text = f"{item.get('lClass', '')} {item.get('facilName', '')} {item.get('mClass', '')}"
                vector = embedding.embed_query(text)
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': item
                }
                es_client.update(index='university_major', id=existing_doc['_id'], body={'doc': doc})

# 업데이트(임베딩 및 인덱스)
def update_major_details():
    major_data = fetch_university_major()
    for item in major_data['dataSearch']['content']:
        major_seq = item['majorSeq']
        major_details = fetch_major_details(major_seq)
        if 'dataSearch' in major_details and 'content' in major_details['dataSearch'] and major_details['dataSearch']['content']:
            detail_item = major_details['dataSearch']['content'][0]
            query = {
                "query": {
                    "match": {
                        "metadata.major": detail_item['major']
                    }
                }
            }
            result = es_client.search(index="major_details", body=query)

            if result['hits']['total']['value'] == 0:
                # 새로운 데이터 추가
                text = f"{detail_item.get('major', '')} {detail_item.get('summary', '')}"
                vector = embedding.embed_query(text)
                doc = {
                    'text': text,
                    'vector': vector,
                    'metadata': detail_item
                }
                es_client.index(index='major_details', body=doc)
            else:
                # 기존 데이터 업데이트
                existing_doc = result['hits']['hits'][0]
                if existing_doc['_source']['metadata'] != detail_item:
                    text = f"{detail_item.get('major', '')} {detail_item.get('summary', '')}"
                    vector = embedding.embed_query(text)
                    doc = {
                        'text': text,
                        'vector': vector,
                        'metadata': detail_item
                    }
                    es_client.update(index='major_details', id=existing_doc['_id'], body={'doc': doc})

# 업데이트(임베딩 및 인덱스)
def update_pdf_data():
    # PDF 데이터는 일반적으로 자주 변경되지 않으므로, 
    # 파일이 변경된 경우에만 업데이트하는 로직을 구현할 수 있습니다.
    # 예를 들어, 파일의 수정 날짜를 확인하여 변경된 경우에만 처리할 수 있습니다.
    pass

# 멀티 서치를 하는 코드
async def multi_index_search(query, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=10):
    if isinstance(query, dict):
        query = query.get('question', '')
    
    entities = extract_entities(query)
    query_vector = embedding.embed_query(query)
    es_query = generate_elasticsearch_query(entities)
    
    index_weights = {
        'university_data': 0.4,
        'university_major': 0.3,
        'major_details': 0.2,
        'pdf_data': 0.1
    }

    multi_search_body = []
    for index in indices:
        search_body = {
            "size": top_k * 2,  # Increase initial results for reranking
            "query": {
                "function_score": {
                    "query": es_query["query"],
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "source": f"cosineSimilarity(params.query_vector, 'vector') * {index_weights.get(index, 1.0)} + 1.0",
                                    "params": {"query_vector": query_vector}
                                }
                            }
                        }
                    ],
                    "boost_mode": "multiply"
                }
            },
            "_source": ["text", "metadata"]
        }
        multi_search_body.extend([{"index": index}, search_body])

    results = es_client.msearch(body=multi_search_body)

    processed_results = []
    for i, response in enumerate(results['responses']):
        if response['hits']['hits']:
            for hit in response['hits']['hits']:
                processed_results.append({
                    'index': indices[i],
                    'score': hit['_score'],
                    'text': hit['_source']['text'],
                    'metadata': hit['_source']['metadata']
                })

    # Reranking using CrossEncoder
    if processed_results:
        rerank_input = [(query, result['text']) for result in processed_results]
        rerank_scores = cross_encoder.predict(rerank_input)
        
        for i, score in enumerate(rerank_scores):
            processed_results[i]['rerank_score'] = score

        # Sort by rerank score
        processed_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    return processed_results[:top_k]

def initialize_agent():
    class FunctionRetriever(BaseRetriever):
        async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
            if isinstance(query, dict):
                query = query.get('question', '')
            results = await multi_index_search(query, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=10)
            return [Document(page_content=result['text'][:500], metadata={**result['metadata'], 'rerank_score': result.get('rerank_score', 0)}) for result in results]

        async def get_relevant_documents(self, query: str) -> List[Document]:
            return await self._aget_relevant_documents(query)
        
    retriever = FunctionRetriever()

    # 사용자 정의 프롬프트 템플릿 생성
    prompt_template = """
    You are a Korean university information expert. Your role is to provide accurate and detailed answers to questions about Korean universities using the provided tools.

    Information Provision:
    - Answer questions regarding university admission procedures, programs, majors, and related information.
    - Focus your responses using the extracted entities (universities, majors, keywords).

    Language and Translation:
    - Translate the final response into {language}. Always ensure that the response is translated, and if it is not, make sure to translate it again.
    - Provide only the translated response.

    Structure and Clarity:
    - Present your answers clearly and in an organized manner. Use bullet points or numbered lists if necessary.
    - Include examples or scenarios to illustrate how the information applies.

    Accuracy and Updates:
    - Provide accurate information based on the latest data available from the tools.
    - Advise the user to check official sources for the most current information.

    Extracted Entities:
    - Universities: {universities}
    - Majors: {majors}
    - Keywords: {keywords}

    Use these entities to guide your search and response.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "language", "universities", "majors", "keywords"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x["question"],
            "language": lambda x: x["language"],
            "universities": lambda x: ", ".join(x["universities"]),
            "majors": lambda x: ", ".join(x["majors"]),
            "keywords": lambda x: ", ".join(x["keywords"]),
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | openai
        | StrOutputParser()
    )

    return qa_chain

@app.post("/academic", response_model=Response)
async def query_agent(query: Query):
    # 인덱스 초기화 확인 및 수행
    if not index_exists('university_data') or \
       not index_exists('university_major') or \
       not index_exists('major_details') or \
       not index_exists('pdf_data'):
        print("Initial setup required. Running full indexing process...")
        await embed_and_index_university_data()
        await embed_and_index_university_major()
        await embed_and_index_major_details()
        await embed_and_index_pdf_data()
        print("Initial indexing completed.")

    agent_executor = initialize_agent()
    language = detect_language(query.input)
    korean_lang = korean_language(query.input)
    entities = extract_entities(korean_lang)

    # 세션 기록 가져오기 또는 새로 생성
    chat_history = session_histories.get(query.session_id, [])

    response = await agent_executor.ainvoke({
        "question": query.input,
        "chat_history": chat_history,
        "agent_scratchpad": [],
        "language": language,
        "universities": entities.universities,
        "majors": entities.majors,
        "keywords": entities.keywords
    })

    # 대화 기록 업데이트
    chat_history.append({"role": "user", "content": query.input})
    chat_history.append({"role": "assistant", "content": response})
    session_histories[query.session_id] = chat_history

    return Response(answer=response)

    
    
    # results = multi_index_search(query)
    
    # print(f"검색 쿼리: {query}")
    # for result in results:
    #     print(f"\nIndex: {result['index']}")
    #     print(f"Score: {result['score']}")
    #     print(f"Text: {result['text'][:100]}...")  # 텍스트의 처음 100자만 출력
    #     print(f"Metadata: {result['metadata']}")

    # print(chat_history)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)