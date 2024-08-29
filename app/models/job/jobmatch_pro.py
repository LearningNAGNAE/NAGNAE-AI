from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.vectorstores import FAISS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
from typing import List, Optional, Dict
from pydantic import BaseModel
import json
import uuid
from ...database.db import get_db
from ...database import crud
from sqlalchemy.orm import Session
import uvicorn

# # Load the job location data
# with open('job_location.json', 'r', encoding='utf-8') as f:
#     job_locations = json.load(f)


# logging.basicConfig(level=logging.DEBUG)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
es_client = Elasticsearch([elasticsearch_url])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드 목록
    allow_headers=["*"],  # 허용할 HTTP 헤더 목록
)

# 세션 ID와 chat_his_no 매핑을 위한 딕셔너리
session_chat_mapping: Dict[str, int] = {}

class ChatRequest(BaseModel):
    question: str
    userNo: int
    categoryNo: int
    session_id: Optional[str] = None
    chat_his_no: Optional[int] = None
    is_new_session: Optional[bool] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    chatHisNo: int
    chatHisSeq: int
    detected_language: str

templates = Jinja2Templates(directory="templates")  # html 파일내 동적 콘텐츠 삽입 할 수 있게 해줌(렌더링).

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0
)

# 1. 언어 감지 및 번역 함수
def gpt_detect_language(text: str) -> str:
    """Language Detection Function"""
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = llm.invoke(messages)
    return response.content.strip().lower()

# 한국어로 번역
def korean_language(text: str) -> str:
    """Function for Translating Multilingual Text to Korean"""
    system_prompt = """You are a multilingual translator specializing in Korean translations. Your task is to translate the given text from any language into natural, fluent Korean. Please follow these guidelines:
    1. First, identify the source language of the given text.
    2. Translate the text accurately into Korean, maintaining the original meaning and nuance.
    3. Provide only the translated Korean text, without any additional explanations or information about the translation process.
    4. Use appropriate honorifics and formal/informal language based on the context.
    5. For specialized terms or proper nouns, provide the Korean translation followed by the original term in parentheses where necessary.
    6. If certain terms are commonly used in their original language even in Korean context, keep them in the original language.
    7. Ensure that idiomatic expressions are translated to their Korean equivalents, not literally.
    """
    human_prompt = f"Please translate the following text into Korean: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = llm.invoke(messages)
    return response.content.strip()





# 2. 잡크롤링
# def jobploy_crawler(lang, pages=5):
#     if isinstance(pages, dict):
#         pages = 5
#     elif not isinstance(pages, int):
#         try:
#             pages = int(pages)
#         except ValueError:
#             pages = 5

#     chrome_driver_path = r"C:\chromedriver\chromedriver.exe"

#     chrome_options = Options()
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_argument("--window-size=1920x1080")
#     # chrome_options.add_argument("--headless")

#     service = Service(chrome_driver_path)
#     driver = webdriver.Chrome(service=service, options=chrome_options)

#     results = []

#     try:
#         for page in range(1, pages + 1):
#             url = f"https://www.jobploy.kr/{lang}/recruit?page={page}"
#             driver.get(url)

#             WebDriverWait(driver, 20).until(
#                 EC.presence_of_element_located((By.CLASS_NAME, "content"))
#             )

#             job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

#             for job in job_listings:
#                 title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
#                 link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
#                 pay_element = job.find_element(By.CSS_SELECTOR, "p.pay")
#                 company_name_element = job.find_element(By.CSS_SELECTOR, "span.text-info")
#                 badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")

#                 if len(badge_elements) >= 3:
#                     location_element = badge_elements[0]
#                     task_element = badge_elements[1]
#                     closing_date_element = badge_elements[2]
#                     location = location_element.text
#                     task = task_element.text
#                     closing_date = closing_date_element.text
#                 else:
#                     closing_date = "마감 정보 없음"
#                     location = "위치 정보 없음"
#                     task = "직무 정보 없음"

#                 title = title_element.text
#                 pay = pay_element.text
#                 company_name = company_name_element.text
                
#                 results.append({
#                     "title": title,
#                     "company_name": company_name,
#                     "link": link_element,
#                     "closing_date": closing_date,
#                     "location": location,
#                     "pay": pay,
#                     "task": task,
#                     "language": lang
#                 })

#     finally:
#         driver.quit()
           
#     return results

# # 크롤링 데이터 가져오기 및 파일로 저장
# default_lang = 'ko'  # 기본 언어를 한국어로 설정
# crawled_data = jobploy_crawler(lang=default_lang, pages=5)

# 3. 엔터티 추출 함수 (ChatGPT 기반)
# def extract_entities(query):
#     system_message = """
#     # Role
#     You are an NER (Named Entity Recognition) machine that specializes in extracting entities from text.

#     # Task
#     - Extract the following entities from the user query: LOCATION, MONEY, OCCUPATION, and PAY_TYPE.
#     - Return the extracted entities in a fixed JSON format, as shown below.

#     # Entities
#     - **LOCATION**: Identifies geographical locations (e.g., cities, provinces). 
#     - In Korean, locations often end with "시" (si), "군" (gun), or "구" (gu).
#     - In English or other languages, locations may end with "-si", "-gun", or "-gu".
#     - Ensure "시" is not misinterpreted or separated from the city name.
#     - **Special Case**: "화성" should always be interpreted as "Hwaseong" in South Korea, and never as "Mars". This should override any other interpretation.
#     - **MONEY**: Identify any salary information mentioned in the text. This could be represented in different forms:
#     - Examples include "250만원", "300만 원", "5천만 원" etc.
#     - Convert amounts expressed in "만원" or "천원" to full numerical values. For example:
#         - "250만원" should be interpreted as 250 * 10,000 = 2,500,000원.
#         - "5천만원" should be interpreted as 5,000 * 10,000 = 50,000,000원.
#     - Extract the numerical value in its full form.
#     - **OCCUPATION**: Detects job titles or professions.
#     - **PAY_TYPE**: Identifies the type of payment mentioned. This could be:
#     - "연봉" or "annual salary" for yearly salary
#     - "월급" or "salary" for monthly salary
#     - "시급" or "hourly" for hourly salary

#     # Output Format
#     - The output should be a JSON object with the following structure:
#     {"LOCATION": "", "MONEY": "", "OCCUPATION": "", "PAY_TYPE": ""}

#     # Policy
#     - If there is no relevant information for a specific entity, return null for that entity.
#     - Do not provide any explanations or additional information beyond the JSON output.
#     - The output should be strictly in the JSON format specified.

#     # Examples
#     - Query: "화성에 연봉 3천만원 이상 주는 생산직 일자리 있어?"
#     Output: {'LOCATION': '화성', 'MONEY': '30,000,000', 'OCCUPATION': '생산', 'PAY_TYPE': '연봉'}
#     """

#     messages = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": query}
#     ]

#     # Generate the response from the model
#     response = llm(messages=messages)
    
#     print(f"==============================================")
#     print(response)
#     print(f"==============================================")
#     # Handle response based on its actual structure
#     response_content = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
    


#     # Parse the JSON response
#     try:
#         entities = json.loads(response.content)
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON: {e}")
#         entities = {"LOCATION": None, "MONEY": None, "OCCUPATION": None, "PAY_TYPE": None}
    
#     print("잘뽑아 왔나!!:", entities)
#     return entities

# 4. 벡터 스토어 생성 함수
# def create_faiss_index(data):
#     # 텍스트와 메타데이터 생성
#     texts = [f"{item['title']} {item['company_name']} {item['link']} {item['closing_date']} {item['location']} {item['pay']} {item['task']}"
#              for item in data]
#     metadata = [{k: item[k] for k in ['title', 'company_name', 'link', 'closing_date', 'location', 'pay', 'task']} for item in data]
    
#     # print("임베딩된:", texts)
#     # print(metadata);

#     # 벡터화
#     embeddings = OpenAIEmbeddings()
    
#     # FAISS 인덱스 생성
#     vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    
#     return vectorstore

# faiss_index = create_faiss_index(crawled_data)

# 5. ElasticSearch 인덱스 생성 및 데이터 업로드
# def create_elasticsearch_index(data, index_name="jobs"):
#     # 기존 인덱스 삭제 (있을 경우)
#     if es_client.indices.exists(index=index_name):
#         es_client.indices.delete(index=index_name)

#     # 새로운 매핑 생성
#     mapping = {
#         "mappings": {
#             "properties": {
#                 "title": {"type": "text"},
#                 "company_name": {"type": "text"},
#                 "link": {"type": "text"},
#                 "closing_date": {"type": "text"},
#                 "location": {"type": "text"},
#                 "pay": {"type": "text"},
#                 "pay_amount": {"type": "long"},  # 급여 필드를 정수형으로 추가
#                 "pay_type": {"type": "text"},
#                 "task": {"type": "text"},
#                 "language": {"type": "keyword"}
#             }
#         }
#     }

#     # 인덱스 생성
#     es_client.indices.create(index=index_name, body=mapping)

#     # 데이터 인덱싱
#     for item in data:
#         pay_str = item.get("pay", "")
#         pay_amount = None
#         pay_type = None

#         # 급여 유형 및 금액 처리
#         if "연봉" in pay_str or "annual salary" in pay_str.lower():
#             pay_type = "annual salary"
#         elif "월급" in pay_str or "salary" in pay_str.lower():
#             pay_type = "salary"
#         elif "시급" in pay_str or "hourly" in pay_str.lower():
#             pay_type = "hourly"

        


#         if pay_type:
#             try:
#                 # 숫자 부분만 추출하여 pay_amount로 변환
#                 pay_amount = int(pay_str.split(':')[1].replace('원', '').replace('KRW', '').replace('$', '').replace(',', '').strip())
#             except ValueError:
#                 pass

#         item["pay_type"] = pay_type
#         item["pay_amount"] = pay_amount
        

#         # FAISS용 텍스트 필드 추가
#         item["vector_text"] = f"{item['pay']} {item['task']} {item['location']}"
#         es_client.index(index=index_name, body=item)

#     print(f"Total {len(data)} documents indexed.")

# 6. 언어별 데이터 저장 함수
# def save_data_to_file(data, filename):
#     with open(filename, "w", encoding="utf-8") as file:
#         json.dump(data, file, ensure_ascii=False, indent=2)

# # 크롤링 데이터를 텍스트 파일로 저장
# save_data_to_file(crawled_data, f"crawled_data_{default_lang}.txt")

# # ElasticSearch에 데이터 업로드
# create_elasticsearch_index(crawled_data)

# 7. 검색 도구 함수 정의
@tool
def search_jobs(query: str) -> str:
    """
    Search for jobs based on the given query string.
    
    Args:
        query (str): The search query to find matching jobs.
    
    Returns:
        str: A list of jobs that match the query.
    """
    
    print(f"\n=== 검색 시작: '{query}' ===")
    
    
    entities = extract_entities(query)
    print(f"추출된 엔티티: {json.dumps(entities, ensure_ascii=False, indent=2)}")

    # ElasticSearch 검색
    es_results = es_client.search(index="jobs", body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["pay", "task", "location"]
            }
        },
        "size": 100
    })
    es_hits = es_results['hits']['hits']

    print(f"ElasticSearch 검색 결과 수: {len(es_hits)}")

    # FAISS 검색 (메타데이터 필터링 포함)
    faiss_results = faiss_index.similarity_search(
        query, 
        k=10,
        filter=lambda x: (
            (entities.get("LOCATION", "").lower() in x.get("location", "").lower()) and
            (entities.get("MONEY", "").replace(",", "").isdigit() and int(entities.get("MONEY", "").replace(",", "")) <= int(x.get("pay_amount", 0)))
        )
    )
    print(f"FAISS 검색 결과 수: {len(faiss_results)}")

    # ElasticSearch와 Faiss 검색결과병합 및 필터링
    combined_results = []
    seen_links = set()

    for hit in es_hits:
        job_data = hit['_source']
        if job_data['link'] not in seen_links:
            combined_results.append(job_data)
            seen_links.add(job_data['link'])

    for doc in faiss_results:
        if doc.metadata['link'] not in seen_links:
            combined_results.append(doc.metadata)
            seen_links.add(doc.metadata['link'])
       
    print(f"병합된 검색 결과 수: {len(combined_results)}")

    # 엔티티 기반 필터링(combined_results를 또 필터링)
    filtered_results = []
    for job in combined_results:
        if entities["LOCATION"] and entities["LOCATION"].lower() not in job["location"].lower():
            print(f"LOCATION 필터링됨: {job['location']}")
            continue
        if entities["MONEY"]:
            try:
                required_salary = int(entities["MONEY"].replace('만원', '0000').replace('원', '').replace(',', '').strip())
                job_salary = int(job["pay"].split(':')[1].replace('원', '').replace(',', '').strip())
                if job_salary < required_salary:
                    print(f"MONEY 필터링됨: {job['pay']}")
                    continue
            except ValueError:
                print(f"MONEY 파싱 실패: {job['pay']}")
                pass
        if entities["OCCUPATION"] and entities["OCCUPATION"].lower() not in job["title"].lower() and entities["OCCUPATION"].lower() not in job["task"].lower():
            print(f"OCCUPATION 필터링됨: {job['title']} / {job['task']}")
            continue
        if entities["PAY_TYPE"]:
            pay_type = entities["PAY_TYPE"].lower()
            job_pay_type = job.get("pay_type", "").lower()
            if pay_type in ["연봉", "annual salary"] and job_pay_type not in ["연봉", "annual salary"]:
                print(f"PAY_TYPE 필터링됨: {job['pay_type']}")
                continue
            elif pay_type in ["월급", "salary"] and job_pay_type not in ["월급", "salary"]:
                print(f"PAY_TYPE 필터링됨: {job['pay_type']}")
                continue
            elif pay_type in ["시급", "hourly"] and job_pay_type not in ["시급", "hourly"]:
                print(f"PAY_TYPE 필터링됨: {job['pay_type']}")
                continue
        
        print(f"필터링 후 추가됨: {job}")
        filtered_results.append(job)

    print(f"\n필터링 후 결과 수: {len(filtered_results)}")

    result = json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results,
        "additional_info": "These are the job listings that match your query."
    }, ensure_ascii=False, indent=2)
    
    return result





# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents([
#     Document(page_content=json.dumps(item, ensure_ascii=False)) 
#     for item in crawled_data
# ])

tools = [search_jobs]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a precise search engine operating based on a pre-crawled Korean job database. Your main function is to extract accurate keywords from user queries and provide only the job listings that exactly match those criteria.

        Language and Translation:
        - The user's query language has been detected as {gpt_detect}.
        - Translate the final response into {gpt_detect}.
        - Ensure that all job-related information (titles, descriptions, etc.) is accurately translated.
        - Maintain the original Korean names for locations and companies, but provide translations in parentheses where necessary.
        - For salary information, convert the amounts to the appropriate currency if needed, but also include the original KRW amount.
        - Provide only the translated response, but keep any proper nouns or specific terms in their original form if translation might cause confusion.


        Accurately extract and utilize the following information from the user's query:
        1. LOCATION:
           - In Korean, look for place names ending with "시" (si), "군" (gun), or "구" (gu).
           - In English, look for place names ending with "-si", "-gun", or "-gu".
           - Include only job listings that exactly match the extracted location.

        2. MONEY (Salary):
           - Convert to exact numeric values. Examples:
             "250만원" → 2,500,000 KRW
             "5천만원" → 50,000,000 KRW
           - Include only job listings that offer a salary equal to or greater than the amount specified by the user.
           - Filter based on the correct type of salary (annual, monthly, hourly).

        3. OCCUPATION:
           - Include only job listings that exactly match or are closely related to the occupation keywords.

        Search and Response Guidelines:
        1. Provide only job listings that exactly match all extracted keywords (location, salary, occupation).
        2. Completely exclude any information that does not match one or more of the keywords.
        3. Mention the total number of search results first.
        4. Clearly present the following details for each job listing:
           - Title
           - Company Name (if available)
           - Location
           - Salary Information (exact amount and type)
           - Job Duties
           - Brief Job Description (if available)
           - Application Link
           - Closing Date
        5. If there are no results, clearly state this and suggest adjusting the search criteria.
        6. Additionally, if the user is not satisfied with the search results, recommend other job search platforms such as:
           - [JobPloy](https://www.jobploy.kr/)
           - [JobKorea](https://www.jobkorea.co.kr/)
           - [Saramin](https://www.saramin.co.kr/)
           - [Albamon](https://www.albamon.com/)

        Your responses should be concise and accurate, providing only information that is 100% relevant to the user's query.
        Do not include any information that is irrelevant or only partially matches the criteria.

        Important: Use the maximum token limit available to provide detailed and comprehensive information.
        Include as many relevant job listings as possible, but provide detailed descriptions for each.
        """
    ),
    (
        "human",
        "{input}"
    ),
    (
        "assistant",
        "Understood. I will search for job listings that precisely match your request. I will extract the keywords for location, salary, and occupation, and provide results that fully meet all the conditions. I will strictly exclude any partially matching or unrelated information."
    ),
    # MessagesPlaceholder(variable_name=MEMORY_KEY),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "gpt_detect": lambda x: x["gpt_detect"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 검색 결과를 사용자의 언어로 번역
def translate_to_user_language(text: str, target_language: str) -> str:
    """한국어 텍스트를 사용자의 언어로 번역하는 함수"""
    system_prompt = f"""You are a multilingual translator. Please translate the following Korean text into {target_language}. Ensure the translation is natural and accurate."""
    human_prompt = f"{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = llm.invoke(messages)
    return response.content.strip()



async def search_jobs_endpoint(chat_request: ChatRequest, db: Session = Depends(get_db)):

    question = chat_request.question
    userNo = chat_request.userNo
    categoryNo = chat_request.categoryNo
    session_id = chat_request.session_id or str(uuid.uuid4())
    chat_his_no = chat_request.chat_his_no
    is_new_session = chat_request.is_new_session


    gpt_detect = gpt_detect_language(question)
    ko_language = korean_language(question)


    result = agent_executor.invoke({
        "input": ko_language, 
        "gpt_detect": gpt_detect,
    })
    
     # 결과를 사용자의 언어로 번역
    translated_result = translate_to_user_language(result["output"], gpt_detect)


    # 채팅 기록 저장
    chat_history = crud.create_chat_history(db, userNo, categoryNo, question, translated_result, is_new_session, chat_his_no)
    
    # 세션 ID와 chat_his_no 매핑 업데이트
    session_chat_mapping[session_id] = chat_history.CHAT_HIS_NO

    chat_response = ChatResponse(
            question=question,
            answer=translated_result,
            chatHisNo=chat_history.CHAT_HIS_NO,
            chatHisSeq=chat_history.CHAT_HIS_SEQ,
            detected_language=gpt_detect
        )





    return JSONResponse(content=chat_response.dict())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

