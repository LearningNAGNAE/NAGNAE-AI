import requests
import os, json
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
from typing import List

load_dotenv()

# 환경 변수 로드
university_api_key = os.getenv("UNIVERSITY_API_KEY")
gpt_api_key = os.getenv("OPENAI_API_KEY")
elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\2025학년도 재외국민과 외국인 특별전형 시행계획 주요사항.pdf"

# 일관된 값을 위하여 Temperature 0.1로 설정 model은 gpt-4o로 설정
openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=gpt_api_key, temperature=0.1)

# Elasticsearch 클라이언트 설정
es_client = Elasticsearch([elasticsearch_url])
embedding = OpenAIEmbeddings()

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
def multi_index_search(query, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=100):
    # 쿼리 텍스트를 벡터로 변환
    query_vector = embedding.embed_query(query)

    # 각 인덱스에 대한 가중치 설정
    index_weights = {
        'university_data': 0.4,
        'university_major': 0.3,
        'major_details': 0.2,
        'pdf_data': 0.1
    }

    # 멀티 인덱스 검색 쿼리 구성
    multi_search_body = []
    for index in indices:
        search_body = {
            "size": top_k,
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"text": query}}
                            ]
                        }
                    },
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

    # 멀티 검색 실행
    results = es_client.msearch(body=multi_search_body)

    # 결과 처리
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

    # 결과를 점수 기준으로 정렬
    processed_results.sort(key=lambda x: x['score'], reverse=True)

    return processed_results[:top_k]


def initialize_agent():

    class FunctionRetriever(BaseRetriever):
        def get_relevant_documents(self, query: str) -> List[Document]:
            results = multi_index_search(query, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=100)
            return [Document(page_content=result['text'], metadata=result['metadata']) for result in results]


    retriever = FunctionRetriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "elasticsearch_search",
        "This tool is used to search college admissions, major information, and more in Elasticsearch."
    )

    tools = [retriever_tool]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            """
            You are a Korean university information expert. Your role is to provide accurate and detailed answers to questions about universities in Korea using the provided tools.

            Information Provision:

            Answer questions regarding university admission procedures, programs, majors, and related information.
            Language and Translation:

            Translate the final response into ({language}).
            Provide only the translated response.
            Structure and Clarity:

            Present answers clearly and organized. Use bullet points or numbered lists for details if needed.
            Include examples or scenarios to illustrate how the information applies when relevant.
            Accuracy and Updates:

            Ensure the information provided is accurate and up-to-date based on the latest data from the tools.
            Advise users to check official sources for the most current information.
            Example Response:

            University Admission Procedures:

            Requirements: [List of required documents and procedures].
            Deadlines: [Provide application submission deadlines].
            Program Details:

            Majors Available: [List of available majors and specializations].
            Curriculum: [Brief overview of the curriculum for the chosen program].
            Translation:

            Translate the above details into the specified language and provide only this translated version.
            Note: Provide only the translated response.
            """
        ),
        ("human", "{input}"),
        
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("system", "Detected language: {language}")
    ])

    agent = create_openai_tools_agent(openai, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor



def main():
    # 임베딩 및 인덱스 생성을 하지 않할지 정하는 코드
    if not index_exists('university_data') or \
       not index_exists('university_major') or \
       not index_exists('major_details') or \
       not index_exists('pdf_data'):
        print("Initial setup required. Running full indexing process...")
        embed_and_index_university_data()
        embed_and_index_university_major()
        embed_and_index_major_details()
        embed_and_index_pdf_data()
        print("Initial indexing completed.")
    else:
        print("Updating existing indices...")
        # update_indices()
        print("Update completed.")


    agent_executor = initialize_agent()

    # 대화 기록을 저장할 리스트를 초기화합니다.
    chat_history = []
    
    while True:
        query = input("질문을 입력하세요 (종료하려면 'q' 입력): ")
        language = detect_language(query)
        korean_lang = korean_language(query)
        print(language);
        print(korean_lang);
        print(f"검색 쿼리: {query}")
        if query.lower() == 'q':
            break

        # 대화 기록을 업데이트하며 에이전트를 실행합니다.
        response = agent_executor.invoke({
            "input": korean_lang,
            "chat_history": chat_history,
            "agent_scratchpad": [],
            "language": language 
        })

        # 응답 출력
        print("응답:", response['output'])
        # chat_history에 새로운 대화 추가
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response['output']})


    
    
    # results = multi_index_search(query)
    
    # print(f"검색 쿼리: {query}")
    # for result in results:
    #     print(f"\nIndex: {result['index']}")
    #     print(f"Score: {result['score']}")
    #     print(f"Text: {result['text'][:100]}...")  # 텍스트의 처음 100자만 출력
    #     print(f"Metadata: {result['metadata']}")

    # print(chat_history)

if __name__ == "__main__":
    main()