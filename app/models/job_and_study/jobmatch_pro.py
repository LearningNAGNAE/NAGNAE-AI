from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
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
import spacy
from dotenv import load_dotenv
from langid import classify
from fastapi.middleware.cors import CORSMiddleware

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 허용할 Origin 목록
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드 목록
    allow_headers=["*"],  # 허용할 HTTP 헤더 목록
)
templates = Jinja2Templates(directory="templates")  # html 파일내 동적 콘텐츠 삽입 할 수 있게 해줌(렌더링).

nlp = spacy.load("ko_core_news_sm")

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1
)

# 언어 감지 함수
def detect_language(text):
    lang, _ = classify(text)
    return lang

# 엔터티 추출 함수
# 현재 스크립트의 디렉토리 경로를 얻습니다
current_dir = os.path.dirname(os.path.abspath(__file__))

# job_location.json 파일 경로를 생성합니다
json_path = os.path.join(current_dir, 'job_location.json')

# job_location.json 파일 로드
with open(json_path, 'r', encoding='utf-8') as f:
    job_locations = json.load(f)

def extract_entities(query):
    doc = nlp(query)
    entities = {
        "LOCATION": [],
        "MONEY": [],
        "OCCUPATION": []
    }
    
    # spaCy의 엔티티 인식 결과 처리
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    words = query.split()
    for word in words:
        # 지역명 처리
        for location, data in job_locations.items():
            # aliases 확인
            if word in data['aliases']:
                entities["LOCATION"].append(location)
                break
            
            # areas 확인
            for area in data['areas']:
                if word == area['ko'] or word == area['en']:
                    entities["LOCATION"].append(f"{location} {area['ko']}")
                    break
            else:
                continue
            break

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents([
        Document(
            page_content=json.dumps(item, ensure_ascii=False),
            metadata={"location": item.get("location", "")}  # 메타데이터에 location 추가
        ) 
        for item in data
    ])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts([text.page_content for text in texts], embeddings, metadatas=[text.metadata for text in texts])

# 글로벌 변수로 벡터 스토어 딕셔너리 선언
vectorstores = {}

# Query processing function to detect language and extract entities
def process_query(query: str):
    lang = detect_language(query)
    entities = extract_entities(query)
    return lang, entities

@tool
def search_jobs(query: str) -> str:
    """Search for job listings based on the given query."""
    lang = detect_language(query)
    if lang not in vectorstores:
        data = load_data_from_file(lang)
        if data is None:
            data = jobploy_crawler(lang)
            save_data_to_file(data, lang)
        vectorstores[lang] = create_vectorstore(data)

    entities = extract_entities(query)

    # 위치 필터링을 위한 람다 함수 정의
    location_filter = None
    if entities["LOCATION"]:
        # 위치 엔터티가 있는 경우에만 필터를 적용
        location_filter = lambda d: any(
            loc.lower() in d.get('location', '').lower()
            for loc in entities["LOCATION"]
        )

    # 유사도 검색을 필터와 함께 실행
    docs = vectorstores[lang].similarity_search(query, k=10)
    
    filtered_results = []

    for doc in docs:
        job_info = json.loads(doc.page_content)
        if location_filter and not location_filter(job_info):
            continue

        match = True  # 조건이 모두 충족되는지 확인하는 플래그

        # 급여 필터링
        if entities["MONEY"]:
            try:
                # 사용자가 입력한 급여 정보 추출 및 정수로 변환
                required_salary_str = entities["MONEY"][0].replace(',', '').replace('원', '').strip()
                required_salary = int(required_salary_str)
                
                # 직무의 급여 정보 추출 및 정수로 변환
                pay_elements = job_info.get('pay', '').split()
                if len(pay_elements) >= 3:
                    job_salary_str = pay_elements[2].replace(',', '').replace('원', '').strip()
                    job_salary = int(job_salary_str)    
                    
                    # 직무의 급여가 요구 급여보다 낮으면 필터링
                    if job_salary < required_salary:
                        continue
                else:
                    # 급여 정보가 부족한 경우 필터링
                    continue
                    
            except ValueError:
                # 급여 정보가 올바르지 않거나 변환 실패 시 필터링
                continue

        # 직무 필터링
        if entities["OCCUPATION"] and match:
            occupation_match = any(
                occ.lower() in job_info.get('title', '').lower() or 
                occ.lower() in job_info.get('task', '').lower() 
                for occ in entities["OCCUPATION"]
            )
            if not occupation_match:
                match = False

        if match:
            filtered_results.append(job_info)

    if not filtered_results:
        return "No job listings found for the specified criteria."

    return json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results,
        "additional_info": f"These are the job listings that match your query."
    }, ensure_ascii=False, indent=2)


# 크롤링 데이터 가져오기 및 파일로 저장
default_lang = 'ko'  # 기본 언어를 한국어로 설정
crawled_data = jobploy_crawler(lang=default_lang, pages=3)

# 크롤링 데이터를 텍스트 파일로 저장
save_data_to_file(crawled_data, f"crawled_data_{default_lang}.txt")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents([
    Document(page_content=json.dumps(item, ensure_ascii=False)) 
    for item in crawled_data
])

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

    detected_lang = detect_language(query)
    if detected_lang not in vectorstores:
        data = load_data_from_file(detected_lang)
        if data is None:
            data = jobploy_crawler(detected_lang)
            save_data_to_file(data, detected_lang)
        vectorstores[detected_lang] = create_vectorstore(data)

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