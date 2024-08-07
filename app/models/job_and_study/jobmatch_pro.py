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
templates = Jinja2Templates(directory="templates")

nlp = spacy.load("ko_core_news_sm")

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1
)

# 언어 감지 함수
def detect_language(text):
    lang, _ = classify(text)
    return lang

def extract_entities(query):
    doc = nlp(query)
    entities = {
        "LOCATION": [],
        "MONEY": [],
        "OCCUPATION": []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    location_keywords = {
        "서울": "서울특별시", "경기": "경기도", "인천": "인천광역시", "부산": "부산광역시", 
        "대구": "대구광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", 
        "세종": "세종특별자치시", "강원": "강원도", "충북": "충청북도", "충남": "충청남도", 
        "전북": "전라북도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", 
        "제주": "제주특별자치도"
    }
    
    major_areas = {
        "수원": "시", "성남": "시", "안양": "시", "안산": "시", "용인": "시", "부천": "시", 
        "광명": "시", "평택": "시", "과천": "시", "오산": "시", "시흥": "시", "군포": "시", 
        "의왕": "시", "하남": "시", "이천": "시", "안성": "시", "김포": "시", "화성": "시", 
        "광주": "시", "여주": "시", "고양": "시", "의정부": "시", "파주": "시", "양주": "시", 
        "구리": "시", "남양주": "시", "포천": "시", "동두천": "시",
        "가평": "군", "양평": "군", "연천": "군",
        "청주": "시", "충주": "시", "제천": "시", "보은": "군", "옥천": "군", "영동": "군", 
        "증평": "군", "진천": "군", "괴산": "군", "음성": "군", "단양": "군",
        "천안": "시", "공주": "시", "보령": "시", "아산": "시", "서산": "시", "논산": "시", 
        "계룡": "시", "당진": "시", "금산": "군", "부여": "군", "서천": "군", "청양": "군", 
        "홍성": "군", "예산": "군", "태안": "군",
        "전주": "시", "군산": "시", "익산": "시", "정읍": "시", "남원": "시", "김제": "시", 
        "완주": "군", "진안": "군", "무주": "군", "장수": "군", "임실": "군", "순창": "군", 
        "고창": "군", "부안": "군",
        "목포": "시", "여수": "시", "순천": "시", "나주": "시", "광양": "시", "담양": "군", 
        "곡성": "군", "구례": "군", "고흥": "군", "보성": "군", "화순": "군", "장흥": "군", 
        "강진": "군", "해남": "군", "영암": "군", "무안": "군", "함평": "군", "영광": "군", 
        "장성": "군", "완도": "군", "진도": "군", "신안": "군",
        "포항": "시", "경주": "시", "김천": "시", "안동": "시", "구미": "시", "영주": "시", 
        "영천": "시", "상주": "시", "문경": "시", "경산": "시", "군위": "군", "의성": "군", 
        "청송": "군", "영양": "군", "영덕": "군", "청도": "군", "고령": "군", "성주": "군", 
        "칠곡": "군", "예천": "군", "봉화": "군", "울진": "군", "울릉": "군",
        "창원": "시", "진주": "시", "통영": "시", "사천": "시", "김해": "시", "밀양": "시", 
        "거제": "시", "양산": "시", "의령": "군", "함안": "군", "창녕": "군", "고성": "군", 
        "남해": "군", "하동": "군", "산청": "군", "함양": "군", "거창": "군", "합천": "군",
        "제주": "시", "서귀포": "시"
    }
    
    words = query.split()
    for word in words:
        if word in location_keywords:
            entities["LOCATION"].append(location_keywords[word])
        elif word.endswith(("시", "군", "구")):
            entities["LOCATION"].append(word)
        elif word in major_areas:
            entities["LOCATION"].append(word + major_areas[word])
    
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

    #languages = ['ko', 'en', 'vi', 'mn', 'uz']
    
    try:
        for page in range(1, pages + 1):
            url = f"https://www.jobploy.kr/{lang}/recruit?page={page}"
            driver.get(url)

            WebDriverWait(driver, 10).until(
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
def save_data_to_file(data, lang):
    filename = f"crawled_data_{lang}.txt"
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
        Document(page_content=json.dumps(item, ensure_ascii=False)) 
        for item in data
    ])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts([text.page_content for text in texts], embeddings)

# 글로벌 변수로 벡터 스토어 딕셔너리 선언
vectorstores = {}

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
    docs = vectorstores[lang].similarity_search(query, k=10)
    filtered_results = []

    for doc in docs:
        job_info = json.loads(doc.page_content)
        match = True  # 조건이 모두 충족되는지 확인하는 플래그

        # 위치 필터링
        if entities["LOCATION"]:
            location_match = any(
                loc.lower() in job_info['location'].lower() or 
                any(part.lower() in job_info['location'].lower() for part in loc.split()) 
                for loc in entities["LOCATION"]
            )
            if not location_match:
                match = False
        else:
            # 사용자가 위치를 명시하지 않은 경우 '서울'을 기본값으로 사용
            if 'Gyeonggi' not in job_info['location']:
                match = False

        # 급여 필터링
        if entities["MONEY"] and match:
            try:
                required_salary_str = entities["MONEY"][0].replace(',', '').replace('원', '').strip()
                required_salary = int(required_salary_str)

                # 3백만원 이상인지 확인
                minimum_salary = 3000000

                pay_elements = job_info['pay'].split()
                if len(pay_elements) >= 3:
                    job_salary_str = pay_elements[2].replace(',', '').replace('원', '').strip()
                    job_salary = int(job_salary_str)    
                    
                    # 급여가 요구 급여 이상인지 확인
                    if job_salary < minimum_salary:
                        match = False
                else:
                    match = False

            except ValueError:
                match = False

        # 직무 필터링
        if entities["OCCUPATION"] and match:
            occupation_match = any(
                occ.lower() in job_info['title'].lower() or 
                occ.lower() in job_info['task'].lower() 
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

# 크롤링 데이터를 파일로 저장하는 함수
def save_data_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# 크롤링 데이터를 텍스트 파일로 저장
save_data_to_file(crawled_data, f"crawled_data_{default_lang}.txt")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents([
    Document(page_content=json.dumps(item, ensure_ascii=False)) 
    for item in crawled_data
])

# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_texts([text.page_content for text in texts], embeddings)

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
            4. Provide a comprehensive summary of the search results.
            5. Offer detailed information about each relevant job listing.
            6. If the keyword or numerical value does not match the user's query, do not provide any other data.

            Include the following information for each job listing:
            - Title
            - Company (if available)
            - Location
            - Task
            - Salary information
            - Brief job description (if available)
            - Key requirements (if available)
            - Application link

            Ensure your response is clear, concise, and directly addresses the user's query.
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
        (
            "user",
            "{input}"
        ),
        (
            "assistant",
            "Thank you for your query. I'll search for job listings based on your request using the search_jobs tool. I'll provide a summary of the results and detailed information about relevant job listings."
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