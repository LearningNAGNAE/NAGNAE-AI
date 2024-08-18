import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain.agents import Tool, AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from jobploy_crawler import jobploy_crawler
from pydantic import BaseModel
from typing import Union
from langchain.tools import StructuredTool
import re
from langid import classify

# Load environment variables
load_dotenv()

class Query(BaseModel):
    query: str

class MinSalaryQuery(BaseModel):
    min_salary: int    

# Initialize Elasticsearch client  채용 정보를 저장하고 검색
elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
es_client = Elasticsearch([elasticsearch_url])

# Initialize OpenAI embeddings and chat model
embeddings = OpenAIEmbeddings()
# Retrieve the API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI client with the API key
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

vector_store = ElasticsearchStore(
    index_name="jobploy_jobs",
    embedding=embeddings,
    es_url=elasticsearch_url,
    distance_strategy="COSINE"  # 또는 "EUCLIDEAN" 또는 "DOT_PRODUCT"
)

# 언어 감지 함수
def detect_language(text):
    lang, _ = classify(text)
    return lang

# 엔터티 추출 함수 (ChatGPT 기반)
def extract_entities(query):
    system_message = """
    You are a specialized AI assistant for extracting entities from text. Your task is to extract the following entities:
    - LOCATION: Identifies geographical locations (e.g., cities, provinces). 
      - In Korean, locations often end with "시" (si), "군" (gun), or "구" (gu).
      - In English or other languages, locations may end with "-si", "-gun", or "-gu".
    - MONEY: Recognizes salary or wage information.  (e.g., '250만원', '3백만원', etc.).
    - OCCUPATION: Detects job titles or professions.

    Please extract these entities from the following text while considering the specific patterns mentioned for LOCATION.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    # Generate the response from the model
    response = llm(messages=messages)
    
    # Handle response based on its actual structure
    response_content = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
    print("CheckPoint: ")
    print(response_content)

    entities = {
        "LOCATION": [],
        "MONEY": [],
        "OCCUPATION": []
    }
    
    # Extract entities from the response content
    for line in response_content.splitlines():
        if "LOCATION:" in line:
            location = line.split("LOCATION:")[-1].strip()
            if location.endswith(("시", "군", "구", "-si", "-gun", "-gu")):
                entities["LOCATION"].append(location)
        elif "MONEY:" in line:
            entities["MONEY"].append(line.split("MONEY:")[-1].strip())
        elif "OCCUPATION:" in line:
            entities["OCCUPATION"].append(line.split("OCCUPATION:")[-1].strip())
    
    return entities


    
    # Print the raw response for debugging
    print("Raw response from model:")
    print(location)
    
    # Initialize the entities dictionary
    entities = {
        "LOCATION": [],
        "MONEY": [],
        "OCCUPATION": []
    }
    
    # Parsing the response based on the expected format
    for line in response_content.splitlines():
        if line.startswith("LOCATION:"):
            entities["LOCATION"] = [loc.strip() for loc in line.split("LOCATION:")[1].strip().strip("[]").split(",")]
        elif line.startswith("MONEY:"):
            entities["MONEY"] = [money.strip() for money in line.split("MONEY:")[1].strip().strip("[]").split(",")]
        elif line.startswith("OCCUPATION:"):
            entities["OCCUPATION"] = [occ.strip() for occ in line.split("OCCUPATION:")[1].strip().strip("[]").split(",")]
    
    print(entities);
    return entities

# 검색 결과 필터링 함수
def filter_search_results(results, entities):
    filtered_results = []
    recommendations = []

    for result in results:
        match = True
        if entities['LOCATION'] and entities['LOCATION'][0] not in result[0].metadata['location']:
            match = False
        if entities['OCCUPATION'] and entities['OCCUPATION'][0] not in result[0].metadata['title']:
            match = False
        if match:
            filtered_results.append(result)
        else:
            recommendations.append(result)
    
    return filtered_results, recommendations

def parse_pay(pay_str):
    # 시급, 월급, 연봉 구분
    pay_match = re.match(r"(\D+)\s*([\d,]+)\s*원", pay_str)
    if not pay_match:
        return None, None
    pay_type, pay_amount_str = pay_match.groups()
    pay_amount = int(pay_amount_str.replace(",", ""))

    # 단위에 따라 월급, 시급, 연봉 등을 구분
    if "시급" in pay_type:
        pay_type = "hourly"
    elif "월급" in pay_type:
        pay_type = "monthly"
    elif "연봉" in pay_type:
        pay_type = "annual"
    else:
        pay_type = "other"

    return pay_type, pay_amount

def index_job_data():
    job_data = jobploy_crawler('ko', pages=1)

    def process_batch(batch):
        texts = [f"{job['title']} - {job['location']} - {job['pay']} - {job['task']}" for job in batch]
        vectors = [embeddings.embed_query(text) for text in texts]
        metadatas = []

        for job in batch:
            pay_type, pay_amount = parse_pay(job['pay'])
            metadata = {
                'title': job['title'],
                'location': job['location'],
                'pay': job['pay'],
                'pay_type': pay_type,
                'pay_amount': pay_amount,
                'task': job['task'],
                'link': job['link']
            }
            metadatas.append(metadata)
        
        for vector, metadata in zip(vectors, metadatas):
            doc = {
                'text': metadata['title'],
                'vector': vector,
                'metadata': metadata
            }
            es_client.index(index='jobploy_jobs', body=doc)

    process_batch(job_data)

# Ensure the index exists and data is indexed
if es_client.indices.exists(index="jobploy_jobs"):
    es_client.indices.delete(index="jobploy_jobs")
index_job_data()

# Define search function
def search_jobs(query: Union[str, Query, dict]) -> str:
    if isinstance(query, Query):
        query_str = query.query
    elif isinstance(query, str):
        query_str = query
    elif isinstance(query, dict):
        query_str = query.get('query', '')
    else:
        raise ValueError(f"Invalid query type: {type(query)}. Expected str, Query object, or dict.")
    
    results = vector_store.similarity_search_with_score(query_str, k=5)
    
    # 엔터티 추출
    entities = extract_entities(query_str)
    print("=== 엔터티 추출 결과 ===")
    print(entities)
    
    # 검색 결과 필터링
    filtered_results, recommendations = filter_search_results(results, entities)
    print("=== 필터링된 결과 ===")
    for r in filtered_results:
        print(f"Title: {r[0].metadata['title']}, Location: {r[0].metadata['location']}, Pay: {r[0].metadata['pay']}")
    
    print("=== 추천 리스트 ===")
    for r in recommendations:
        print(f"Title: {r[0].metadata['title']}, Location: {r[0].metadata['location']}, Pay: {r[0].metadata['pay']}")
    
    # 필터링된 결과를 기반으로 응답 생성
    filtered_results_str = "\n".join([f"Title: {r[0].metadata['title']}\nLocation: {r[0].metadata['location']}\nPay: {r[0].metadata['pay']}\nTask: {r[0].metadata['task']}\nLink: {r[0].metadata['link']}\nScore: {r[1]}\n" for r in filtered_results])
    
    if recommendations:
        recommendations_str = "\n\nRecommendations:\n" + "\n".join([f"Title: {r[0].metadata['title']}\nLocation: {r[0].metadata['location']}\nPay: {r[0].metadata['pay']}\nTask: {r[0].metadata['task']}\nLink: {r[0].metadata['link']}\nScore: {r[1]}\n" for r in recommendations])
    else:
        recommendations_str = ""
    
    return filtered_results_str + recommendations_str

# Define search function by minimum salary
def search_jobs_with_min_salary(min_salary: int) -> str:
    query_body = {
        "query": {
            "range": {
                "metadata.pay_amount": {
                    "gte": min_salary
                }
            }
        }
    }

    results = es_client.search(index='jobploy_jobs', body=query_body)
    hits = results['hits']['hits']
    return "\n".join([f"Title: {hit['_source']['metadata']['title']}\nLocation: {hit['_source']['metadata']['location']}\nPay: {hit['_source']['metadata']['pay']}\nTask: {hit['_source']['metadata']['task']}\nLink: {hit['_source']['metadata']['link']}\n" for hit in hits])

# 엔터티 추출 디버깅
def debug_entities(query: str):
    print("=== 디버깅: 사용자 입력 ===")
    print(query)
    
    # 추출된 엔터티를 확인
    extracted_entities = extract_entities(query)
    print("=== 디버깅: 추출된 엔터티 ===")
    print(extracted_entities)
    return extracted_entities

# 쿼리 생성 및 검색 결과 디버깅
def debug_search_results(query_str: str):
    print("=== 디버깅: 생성된 검색 쿼리 ===")
    print(query_str)
    
    results = vector_store.similarity_search_with_score(query_str, k=5)
    
    print("=== 디버깅: 검색 결과 ===")
    for r in results:
        print(f"Title: {r[0].metadata['title']}, Location: {r[0].metadata['location']}, Pay: {r[0].metadata['pay']}")
    
    return results

# 사용자 요청 처리 과정에서 디버깅 함수 호출
def process_user_request(user_input):
    # 1단계: 엔터티 추출 확인
    entities = debug_entities(user_input)
    
    # 2단계: 검색 쿼리 생성 및 검색 결과 확인
    query_str = entities['LOCATION'] + " " + entities['TASK']  # 예시로 LOCATION과 TASK를 조합하여 쿼리 생성
    results = debug_search_results(query_str)
    
    # 최종적으로 사용자가 원하는 결과 반환
    return results


# Create a tool for the agent to use
tools = [
    StructuredTool.from_function(
        func=search_jobs,
        name="JobSearch",
        description="Useful for searching job listings based on various criteria like location, salary, or job type.",
        args_schema=Query
    ),
    StructuredTool.from_function(
        func=search_jobs_with_min_salary,
        name="JobSearchWithMinSalary",
        description="Useful for finding job listings with a minimum salary threshold.",
        args_schema=MinSalaryQuery
    )
]

# Create the agent
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a specialized AI assistant for job searching and entity extraction. Your primary tasks are:

1. Help users find relevant job listings based on their queries using the JobSearch tool.

2. Extract and analyze key information from job listings and user queries, focusing on these entities:
   - **LOCATION**: Identify geographical locations (e.g., cities, provinces). Be precise in distinguishing between different administrative levels in South Korea:
     - "시" (si), "군" (gun), "구" (gu), "도" (do), and "읍" (eup) are common suffixes for cities, counties, districts, provinces, and towns, respectively.
     - Ensure to capture both the specific city and broader region (e.g., "서울특별시 강남구").
     - For locations written in English, ensure proper translation to or from Korean if necessary.
   - **MONEY**: Recognize salary or wage information and categorize it appropriately:
     - "시급" (hourly wage): Typically shown as 원/시간.
     - "월급" (monthly salary): Typically shown as 원/월.
     - "연봉" (annual salary): Typically shown as 원/년.
     - Clearly specify the type of compensation (hourly, monthly, or annual) when providing job details.
   - **OCCUPATION**: Detect job titles or professions from the "title" field of the job data. Provide a clear and concise summary of the job title.
   - **TASK**: Identify and summarize the main tasks or duties associated with the job position from the "task" field.
   - **LINK**: Include a link to the full job description for more details from the "link" field.
   - **CLOSING_DATE**: Recognize and inform the user about the closing date of the job listing, if available, which indicates how long the listing will remain open.

When responding to user queries:
1. First, use the JobSearch tool to find relevant job listings.
2. Then, analyze the results and the user's query to extract the key entities (LOCATION, MONEY, OCCUPATION, TASK, LINK, CLOSING_DATE).
3. Provide a summary of the job listings, highlighting the extracted entities:
   - Mention the specific type of salary or wage (시급, 월급, 연봉) and ensure clarity on the amount and period.
   - Clearly distinguish between cities, districts, and provinces to avoid confusion in the LOCATION information.
4. If the user's query lacks specificity, ask follow-up questions to refine the search based on location, salary expectations, specific job titles, or task-related keywords.

Always strive to provide accurate, relevant, and helpful information to assist users in their job search process."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Main interaction loop
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        response = agent_executor.run(user_input)
        print(f"Assistant: {response}")