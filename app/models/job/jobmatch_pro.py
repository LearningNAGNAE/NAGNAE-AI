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
- Extract the following entities from the user query: LOCATION, MONEY, and OCCUPATION.
- Return the extracted entities in a fixed JSON format, as shown below.

# Entities
- **LOCATION**: Identifies geographical locations (e.g., cities, provinces). 
  - In Korean, locations often end with "시" (si), "군" (gun), or "구" (gu).
  - In English or other languages, locations may end with "-si", "-gun", or "-gu".
- **MONEY**: Identify any salary or wage information mentioned in the text. This could be represented in different forms:
  - Examples include "250만원", "300만 원", "5천만 원" etc.
  - Convert amounts expressed in "만원" or "천원" to full numerical values. For example:
    - "250만원" should be interpreted as 250 * 10,000 = 2,500,000원.
    - "5천만원" should be interpreted as 5,000 * 10,000 = 50,000,000원.
  - Extract the numerical value in its full form.
- **OCCUPATION**: Detects job titles or professions.

# Output Format
- The output should be a JSON object with the following structure:
  {"LOCATION": "", "MONEY": "", "OCCUPATION": ""}

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
        entities = {"LOCATION": None, "MONEY": None, "OCCUPATION": None}
    
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
                
                results.append({
                    "title": title,
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
    for item in data:
        es_client.index(index=index_name, body=item)

# 언어별 데이터 저장 함수
def save_data_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


@tool
def search_jobs(query: str) -> str:
    """Search for job listings based on the given query."""
    lang = detect_language(query)
    entities = extract_entities(query)

    # 기본적으로 bool 쿼리를 설정합니다.
    search_body = {
        "query": {
            "bool": {
                "must": [],
                "filter": []
            }
        }
    }

    if entities["LOCATION"]:
        # 여기를 수정합니다
        search_body["query"]["bool"]["must"].append({
            "match": {"location": " ".join(entities["LOCATION"])}
        })

    if entities["MONEY"]:
        try:
            # 괄호 안의 숫자 추출
            money_str = entities["MONEY"][0]
            if "(" in money_str:
                required_salary = int(money_str.split('(')[-1].replace('만원', '0000').replace('원', '').replace(',', '').replace(')', '').strip())
            else:
                required_salary = int(money_str.replace('만원', '0000').replace('원', '').replace(',', '').strip())
            
            search_body["query"]["bool"]["filter"].append({
                "range": {"pay": {"gte": required_salary}}
            })
        except ValueError as e:
            print(f"Error processing MONEY entity: {e}")

    if entities["OCCUPATION"]:
        search_body["query"]["bool"]["must"].append({
            "multi_match": {
                "query": " ".join(entities["OCCUPATION"]),
                "fields": ["title", "task"],
                "type": "best_fields"
            }
        })
    # logging.debug(f"Search body: {json.dumps(search_body, ensure_ascii=False, indent=2)}")


    print("인덱스 존재 여부:")
    print(es_client.indices.exists(index="jobs"))

    print("인덱스 내 문서 수:")
    print(es_client.count(index="jobs"))

    print("인덱스 매핑:")
    mapping_response = es_client.indices.get_mapping(index="jobs")
    print(json.dumps(mapping_response.body, indent=2))





    # Elasticsearch로 쿼리 실행
    # Elasticsearch로 쿼리 실행
    print("검색중...")
    res: ObjectApiResponse = es_client.search(index="jobs", body=search_body)
    print(f"Query executed. Response type: {type(res)}")


    # res_dict 데이터 값이 제대로 안들어옴 여기서부터 확인해보면 됩니다.!!!!!!!!!!!!!! res.body에서 값이 제대로 저장 되어 있는지 index안에 저장되어있는 형식에 맞쳐서 값을 가져와 저장을 해야함
    # ObjectApiResponse 객체를 dict로 변환
    print("Converting ObjectApiResponse to dict...")
    res_dict = res.body
    print(f"Conversion complete. res_dict type: {type(res_dict)}")


    if not res_dict['hits']['hits']:
        print("No job listings found.")
        return "No job listings found for the specified criteria."

    print("Processing search results...")
    filtered_results = []
    for hit in res_dict['hits']['hits']:
        filtered_results.append(hit["_source"])
        print(f"Processed job: {hit['_source'].get('title', 'No title')}")

    print(f"Total processed results: {len(filtered_results)}")

    result = json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results,
        "additional_info": "These are the job listings that match your query."
    }, ensure_ascii=False, indent=2)

    print("JSON response created.")
    print(f"Response length: {len(result)}")

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
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a specialized AI assistant focused on job searches in Korea. Your primary function is to provide accurate, relevant, and up-to-date job information based on user queries.

            You utilize the following NER categories to extract key information from user queries:
            - LOCATION: Identifies geographical locations (e.g., cities, provinces)
            - MONEY: Recognizes salary or wage information
            - OCCUPATION: Detects job titles or professions

            When responding to queries:
            1. Analyze user queries by language type to extract relevant entities.
            2. Search the job database using the extracted information.
            3. Filter and prioritize job listings based on the user's requirements.
            4. If the query includes a Location keyword, ensure that all relevant job listings for that location are retrieved and included in the response. I will include all relevant job listings from that location and will not provide information about other locations.
            5. Provide a comprehensive summary of the search results.
            6. Offer detailed information about each relevant job listing.
            7. Filter job listings based on the type of salary information requested by the user:
                - If the user asks for "monthly salary," only include jobs with monthly salary information (labeled as "Salary").
                - If the user asks for "annual salary," only include jobs with annual salary information (labeled as "Annual Salary").
                - If the user asks for "hourly wage," only include jobs with hourly wage information (labeled as "Hourly").
            8. If the keyword or numerical value does not match the user's query, do not provide any other data.

            Include the following information for each job listing:
            - Title
            - Company (if available)
            - Location
            - Task
            - Salary information
            - Brief job description (if available)
            - Key requirements (if available)
            - Application link

            Ensure your response is clear, concise, and directly addresses the user's query. If a LOCATION is mentioned in the query, include all relevant job listings from that location.
            """
        ),
        (
            "user",
            "I need you to search for job listings based on my query. Can you help me with that?"
        ),
        (
            "assistant",
            "Certainly! I'd be happy to help you search for job listings based on your query. Please provide me with your specific request, and I'll use the search_jobs tool to find relevant information for you. What kind of job or criteria are you looking for?"
        ),
        # Few-shot Example 1
        (
            "user",
            "I'm looking for a kitchen job in Seoul with a salary of at least 3,000,000 KRW."
        ),
        (
            "assistant",
            """
            Thank you for your query. I'll search for kitchen job listings in Seoul with a salary of at least 3,000,000 KRW. I'll provide a summary of the results and detailed information about relevant job listings.
            """
        ),
        # Few-shot Example 2
        (
            "user",
            "Can you tell me about kitchen jobs in Hwaseong with an annual salary of more than 3,000,000 KRW?"
        ),
        (
            "assistant",
            """
            Thank you for your query. I'll search for jobs with an annual salary of more than 30,000,000 KRW in Hwaseong. I'll provide a summary of the results and detailed information about relevant job listings.
            """
        ),
        # Few-shot Example 3
        (
            "user",
            "경기도 화성에서 월급 3백만원 이상인 일자리 알려줘"
        ),
        (
            "assistant",
            """
            질문 주셔서 감사합니다. 경기도 화성에서 월급 3,000,000 원 이상의 일자리를 검색해 드리겠습니다. 결과를 요약해서 제공하고, 관련된 구체적인 구인 정보도 함께 안내해 드리겠습니다.
            """
        ),

        # User's query input
        (
            "user",
            "{input}"
        ),
        (
            "assistant",
            "Thank you for your query. I'll search for job listings based on your request using the search_jobs tool. I'll provide a summary of the results and detailed information about relevant job listings.If you mentioned a specific location, I'll ensure to include all relevant job listings from that location."
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/search_jobs")
async def search_jobs_endpoint(request: Request, query: str = Form(...)):
    chat_history = []

    result = agent_executor.invoke({"input": query, "chat_history": chat_history})
    chat_history.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["output"]},
        ]
    )

    return JSONResponse(content={"response": result["output"], "chat_history": chat_history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)