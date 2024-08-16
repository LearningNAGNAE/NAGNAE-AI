import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv

# 중복 라이브러리 오류를 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1
)

# 언어 감지 함수
def detect_language(text):
    lang = "ko"  # 기본적으로 한국어로 설정
    return lang

# JobPloy 크롤러
def jobploy_crawler(lang, pages=3):
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")

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

# 언어별 데이터 저장 및 로드 함수
def save_data_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def load_data_from_file(lang):
    filename = f"crawled_data_{lang}.txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None

# 벡터 스토어 생성 함수
def create_vectorstore(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents([
        Document(
            page_content=json.dumps(item, ensure_ascii=False),
            metadata={"location": item.get("location", "")}
        ) 
        for item in data
    ])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts([text.page_content for text in texts], embeddings, metadatas=[text.metadata for text in texts])

# 글로벌 변수로 벡터 스토어 딕셔너리 선언
vectorstores = {}

# 각 검색 함수에 필요한 매개변수 스키마 정의
class LocationQuery(BaseModel):
    query: str
    location: str

class SalaryQuery(BaseModel):
    query: str
    min_salary: int

class OccupationQuery(BaseModel):
    query: str
    occupation: str

# 특정 위치에 대한 검색 처리 함수
def search_jobs_by_location(query: str, location: str, top_k: int = 10) -> str:
    lang = detect_language(query)
    if lang not in vectorstores:
        data = load_data_from_file(lang)
        if data is None:
            data = jobploy_crawler(lang)
            save_data_to_file(data, lang)
        vectorstores[lang] = create_vectorstore(data)

    docs = vectorstores[lang].similarity_search(query, k=top_k)

    filtered_results = [
        json.loads(doc.page_content)
        for doc in docs
        if location.lower() in json.loads(doc.page_content).get('location', '').lower()
    ]

    if not filtered_results:
        return "No job listings found for the specified location."

    return json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results
    }, ensure_ascii=False, indent=2)

# 특정 급여 기준에 대한 검색 처리 함수
def search_jobs_by_salary(query: str, min_salary: int, top_k: int = 10) -> str:
    lang = detect_language(query)
    if lang not in vectorstores:
        data = load_data_from_file(lang)
        if data is None:
            data = jobploy_crawler(lang)
            save_data_to_file(data, lang)
        vectorstores[lang] = create_vectorstore(data)

    docs = vectorstores[lang].similarity_search(query, k=top_k)

    filtered_results = []
    for doc in docs:
        job_info = json.loads(doc.page_content)
        try:
            pay_elements = job_info.get('pay', '').split()
            if len(pay_elements) >= 3:
                job_salary_str = pay_elements[2].replace(',', '').replace('원', '').strip()
                job_salary = int(job_salary_str)
                if job_salary >= min_salary:
                    filtered_results.append(job_info)
        except ValueError:
            continue

    if not filtered_results:
        return "No job listings found for the specified salary."

    return json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results
    }, ensure_ascii=False, indent=2)

# 특정 직무에 대한 검색 처리 함수
def search_jobs_by_occupation(query: str, occupation: str, top_k: int = 10) -> str:
    lang = detect_language(query)
    if lang not in vectorstores:
        data = load_data_from_file(lang)
        if data is None:
            data = jobploy_crawler(lang)
            save_data_to_file(data, lang)
        vectorstores[lang] = create_vectorstore(data)

    docs = vectorstores[lang].similarity_search(query, k=top_k)

    filtered_results = [
        json.loads(doc.page_content)
        for doc in docs
        if occupation.lower() in (json.loads(doc.page_content).get('title', '').lower() +
                                  json.loads(doc.page_content).get('task', '').lower())
    ]

    if not filtered_results:
        return "No job listings found for the specified occupation."

    return json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results
    }, ensure_ascii=False, indent=2)


# 도구들을 올바른 형식으로 정의
tools = [
    Tool(
        name="search_jobs_by_location",
        func=search_jobs_by_location,
        description="Search for jobs in a specific location.",
        args_schema=LocationQuery,
    ),
    Tool(
        name="search_jobs_by_salary",
        func=search_jobs_by_salary,
        description="Search for jobs with a minimum salary.",
        args_schema=SalaryQuery,
    ),
    Tool(
        name="search_jobs_by_occupation",
        func=search_jobs_by_occupation,
        description="Search for jobs in a specific occupation.",
        args_schema=OccupationQuery,
    ),
]

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

agent_executor = AgentExecutor(agent=llm_with_tools, tools=tools, verbose=True)

@app.post("/search_jobs")
async def search_jobs_endpoint(request: Request, query: str = Form(...), location: str = None, min_salary: int = None, occupation: str = None):
    chat_history = []

    # 도구 선택에 따라 분기 처리
    if location:
        result = tools[0].func(query=query, location=location)
    elif min_salary:
        result = tools[1].func(query=query, min_salary=min_salary)
    elif occupation:
        result = tools[2].func(query=query, occupation=occupation)
    else:
        result = "Please specify a filter (location, salary, or occupation)."

    chat_history.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": result},
        ]
    )

    return JSONResponse(content={"response": result, "chat_history": chat_history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)