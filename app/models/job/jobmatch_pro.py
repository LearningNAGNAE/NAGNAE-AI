from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_core.documents import Document
import json
import os
from dotenv import load_dotenv
from langid import classify
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch
from elastic_transport import ObjectApiResponse
import logging
import re
import json

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
    allow_origins=["*"],  # 허용할 Origin 목록
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드 목록
    allow_headers=["*"],  # 허용할 HTTP 헤더 목록
)
templates = Jinja2Templates(directory="templates")  # html 파일내 동적 콘텐츠 삽입 할 수 있게 해줌(렌더링).

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0
)

# 한국어로 번역
# 오늘 프롬프트 손봐야함
def korean_language(text: str) -> str:
    """다국어 텍스트를 한국어로 번역하는 함수"""
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

# gpt 언어감지
def gpt_detect_language(text: str) -> str:
    """언어 감지 함수"""
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = llm.invoke(messages)
    return response.content.strip().lower()


# 언어 감지 함수
def detect_language(text):
    lang, _ = classify(text)
    return lang

# 엔터티 추출 함수 (ChatGPT 기반)
def extract_entities(query):
    system_message = """
    # Role
    You are an NER (Named Entity Recognition) machine that specializes in extracting entities from text.

    # Task
    - Extract the following entities from the user query: LOCATION, MONEY, OCCUPATION, and PAY_TYPE.
    - Return the extracted entities in a fixed JSON format, as shown below.

    # Entities
    - **LOCATION**: Identifies geographical locations (e.g., cities, provinces). 
    - In Korean, locations often end with "시" (si), "군" (gun), or "구" (gu).
    - In English or other languages, locations may end with "-si", "-gun", or "-gu".
    - Ensure "시" is not misinterpreted or separated from the city name.
    - **MONEY**: Identify any salary information mentioned in the text. This could be represented in different forms:
    - Examples include "250만원", "300만 원", "5천만 원" etc.
    - Convert amounts expressed in "만원" or "천원" to full numerical values. For example:
        - "250만원" should be interpreted as 250 * 10,000 = 2,500,000원.
        - "5천만원" should be interpreted as 5,000 * 10,000 = 50,000,000원.
    - Extract the numerical value in its full form.
    - **OCCUPATION**: Detects job titles or professions.
    - **PAY_TYPE**: Identifies the type of payment mentioned. This could be:
    - "연봉" or "annual salary" for yearly salary
    - "월급" or "salary" for monthly salary
    - "시급" or "hourly" for hourly salary

    # Output Format
    - The output should be a JSON object with the following structure:
    {"LOCATION": "", "MONEY": "", "OCCUPATION": "", "PAY_TYPE": ""}

    # Policy
    - If there is no relevant information for a specific entity, return null for that entity.
    - Do not provide any explanations or additional information beyond the JSON output.
    - The output should be strictly in the JSON format specified.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    # Generate the response from the model
    response = llm(messages=messages)
    
    print(f"==============================================")
    print(response)
    print(f"==============================================")
    # Handle response based on its actual structure
    response_content = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
    


    # Parse the JSON response
    try:
        entities = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        entities = {"LOCATION": None, "MONEY": None, "OCCUPATION": None, "PAY_TYPE": None}
    
    print("잘뽑아 왔나!!:", entities)
    return entities

def jobploy_crawler(lang, pages=3):
    if isinstance(pages, dict):
        pages = 3
    elif not isinstance(pages, int):
        try:
            pages = int(pages)
        except ValueError:
            pages = 3

    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")
    # chrome_options.add_argument("--headless")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []

    try:
        for page in range(1, pages + 1):
            url = f"https://www.jobploy.kr/{lang}/recruit?page={page}"
            driver.get(url)

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

            for job in job_listings:
                title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
                link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                pay_element = job.find_element(By.CSS_SELECTOR, "p.pay")
                company_name_element = job.find_element(By.CSS_SELECTOR, "span.text-info")
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")

                if len(badge_elements) >= 3:
                    location_element = badge_elements[0]
                    task_element = badge_elements[1]
                    closing_date_element = badge_elements[2]
                    location = location_element.text
                    task = task_element.text
                    closing_date = closing_date_element.text
                else:
                    closing_date = "마감 정보 없음"
                    location = "위치 정보 없음"
                    task = "직무 정보 없음"

                title = title_element.text
                pay = pay_element.text
                company_name = company_name_element.text
                
                results.append({
                    "title": title,
                    "company_name": company_name,
                    "link": link_element,
                    "closing_date": closing_date,
                    "location": location,
                    "pay": pay,
                    "task": task,
                    "language": lang
                })

    finally:
        driver.quit()
           
    return results


# ElasticSearch 인덱스 생성 및 데이터 업로드
def create_elasticsearch_index(data, index_name="jobs"):
    # 기존 인덱스 삭제 (있을 경우)
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    # 새로운 매핑 생성
    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "company_name": {"type": "text"},
                "link": {"type": "text"},
                "closing_date": {"type": "text"},
                "location": {"type": "text"},
                "pay": {"type": "text"},
                "pay_amount": {"type": "long"},  # 급여 필드를 정수형으로 추가
                "pay_type": {"type": "text"},
                "task": {"type": "text"},
                "language": {"type": "keyword"}
            }
        }
    }

    # 인덱스 생성
    es_client.indices.create(index=index_name, body=mapping)

    # 데이터 인덱싱
    for item in data:
        pay_str = item.get("pay", "")
        pay_amount = None
        pay_type = None

        # 급여 유형 및 금액 처리
        if "연봉" in pay_str or "annual salary" in pay_str.lower():
            pay_type = "annual salary"
        elif "월급" in pay_str or "salary" in pay_str.lower():
            pay_type = "salary"
        elif "시급" in pay_str or "hourly" in pay_str.lower():
            pay_type = "hourly"

        


        if pay_type:
            try:
                # 숫자 부분만 추출하여 pay_amount로 변환
                pay_amount = int(pay_str.split(':')[1].replace('원', '').replace('KRW', '').replace('$', '').replace(',', '').strip())
            except ValueError:
                pass

        item["pay_type"] = pay_type
        item["pay_amount"] = pay_amount
        es_client.index(index=index_name, body=item)

    print(f"Total {len(data)} documents indexed.")

# 언어별 데이터 저장 함수
def save_data_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# 검색 도구 함수 정의
@tool
def search_jobs(query: str) -> str:
    """Search for job listings based on the given query."""
    print(f"\n=== 검색 시작: '{query}' ===")
    
    lang = detect_language(query)
    
    entities = extract_entities(query)
    print(f"추출된 엔티티: {json.dumps(entities, ensure_ascii=False, indent=2)}")

    # 모든 job 데이터를 가져옵니다.
    all_jobs = es_client.search(index="jobs", body={"query": {"match_all": {}}, "size": 10000})
    print(f"총 검색된 문서 수: {len(all_jobs['hits']['hits'])}")
    
    filtered_results = []

    
    for i, job in enumerate(all_jobs['hits']['hits']):
        job_data = job['_source']
        if i < 5:  # 처음 5개의 문서만 출력
            
            print(f"\n문서 {i+1}:")
            print(json.dumps(job['_source'], ensure_ascii=False, indent=2))
        
        # LOCATION 필터링
        if entities["LOCATION"] and entities["LOCATION"].lower() not in job_data["location"].lower():
            continue

        # MONEY 필터링
        if entities["MONEY"]:
            try:
                required_salary = int(entities["MONEY"].replace('만원', '0000').replace('원', '').replace(',', '').strip())
                job_salary = int(job_data["pay"].split(':')[1].replace('원', '').replace(',', '').strip())
                if job_salary < required_salary:
                    continue
            except ValueError:
                print(f"급여 정보 파싱 실패: {job_data['pay']}")
                pass  # 급여 정보를 파싱할 수 없는 경우 건너뜁니다.

        # OCCUPATION 필터링
        if entities["OCCUPATION"] and entities["OCCUPATION"].lower() not in job_data["title"].lower() and entities["OCCUPATION"].lower() not in job_data["task"].lower():
            continue

        # PAY_TYPE 필터링
        if entities["PAY_TYPE"]:
            pay_type = entities["PAY_TYPE"].lower()
            job_pay_type = job_data.get("pay_type", "").lower()
            
            if pay_type in ["연봉", "annual salary"] and job_pay_type not in ["연봉", "annual salary"]:
                continue
            elif pay_type in ["월급", "salary"] and job_pay_type not in ["월급", "salary"]:
                continue
            elif pay_type in ["시급", "hourly"] and job_pay_type not in ["시급", "hourly"]:
                continue
                
        filtered_results.append(job_data)
        
        if i % 1000 == 0:
            print(f"처리 중: {i+1}/{len(all_jobs['hits']['hits'])} 문서 검사 완료")

    print(f"\n필터링 후 결과 수: {len(filtered_results)}")

    result = json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results,
        "additional_info": "These are the job listings that match your query."
    }, ensure_ascii=False, indent=2)
    
    print("\n=== 검색 결과 요약 ===")
    print(json.dumps({
        "total_jobs_found": len(filtered_results),
        "sample_results": filtered_results[:3] if filtered_results else []
    }, ensure_ascii=False, indent=2))

    return result

# 크롤링 데이터 가져오기 및 파일로 저장
default_lang = 'ko'  # 기본 언어를 한국어로 설정
crawled_data = jobploy_crawler(lang=default_lang, pages=3)

# 크롤링 데이터를 텍스트 파일로 저장
save_data_to_file(crawled_data, f"crawled_data_{default_lang}.txt")

# ElasticSearch에 데이터 업로드
create_elasticsearch_index(crawled_data)


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents([
#     Document(page_content=json.dumps(item, ensure_ascii=False)) 
#     for item in crawled_data
# ])

tools = [search_jobs]

MEMORY_KEY = "chat_history"
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
    MessagesPlaceholder(variable_name=MEMORY_KEY),
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
        "chat_history": lambda x: x["chat_history"],
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


@app.post("/search_jobs")
async def search_jobs_endpoint(request: Request, query: str = Form(...)):
    chat_history = []

    gpt_detect = gpt_detect_language(query)
    ko_language = korean_language(query)


    result = agent_executor.invoke({"input": ko_language, "chat_history": chat_history, "gpt_detect": gpt_detect})
    chat_history.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["output"]},
        ]
    )
    
     # 결과를 사용자의 언어로 번역
    translated_result = translate_to_user_language(result["output"], gpt_detect)

    return JSONResponse(content={"response": translated_result, "chat_history": chat_history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)