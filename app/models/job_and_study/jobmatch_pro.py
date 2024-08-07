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
templates = Jinja2Templates(directory="templates") #html 파일내 동적 콘텐츠 삽입 할 수 있게 해줌(렌더링).

nlp = spacy.load("ko_core_news_sm")

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1
)

# 언어 감지 함수
def detect_language(text):
    lang, _ = classify(text)
    return lang

# 엔터티 추출 함수
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
        "수원": "수원시", "성남": "성남시", "안양": "안양시", "안산": "안산시", "용인": "용인시", 
        "부천": "부천시", "광명": "광명시", "평택": "평택시", "과천": "과천시", "오산": "오산시", 
        "시흥": "시흥시", "군포": "군포시", "의왕": "의왕시", "하남": "하남시", "이천": "이천시", 
        "안성": "안성시", "김포": "김포시", "화성": "화성시", "광주": "광주시", "여주": "여주시", 
        "고양": "고양시", "의정부": "의정부시", "파주": "파주시", "양주": "양주시", "구리": "구리시", 
        "남양주": "남양주시", "포천": "포천시", "동두천": "동두천시", "가평": "가평군", "양평": "양평군", 
        "연천": "연천군", "청주": "청주시", "충주": "충주시", "제천": "제천시", "보은": "보은군", 
        "옥천": "옥천군", "영동": "영동군", "증평": "증평군", "진천": "진천군", "괴산": "괴산군", 
        "음성": "음성군", "단양": "단양군", "천안": "천안시", "공주": "공주시", "보령": "보령시", 
        "아산": "아산시", "서산": "서산시", "논산": "논산시", "계룡": "계룡시", "당진": "당진시", 
        "금산": "금산군", "부여": "부여군", "서천": "서천군", "청양": "청양군", "홍성": "홍성군", 
        "예산": "예산군", "태안": "태안군", "전주": "전주시", "군산": "군산시", "익산": "익산시", 
        "정읍": "정읍시", "남원": "남원시", "김제": "김제시", "완주": "완주군", "진안": "진안군", 
        "무주": "무주군", "장수": "장수군", "임실": "임실군", "순창": "순창군", "고창": "고창군", 
        "부안": "부안군", "목포": "목포시", "여수": "여수시", "순천": "순천시", "나주": "나주시", 
        "광양": "광양시", "담양": "담양군", "곡성": "곡성군", "구례": "구례군", "고흥": "고흥군", 
        "보성": "보성군", "화순": "화순군", "장흥": "장흥군", "강진": "강진군", "해남": "해남군", 
        "영암": "영암군", "무안": "무안군", "함평": "함평군", "영광": "영광군", "장성": "장성군", 
        "완도": "완도군", "진도": "진도군", "신안": "신안군", "포항": "포항시", "경주": "경주시", 
        "김천": "김천시", "안동": "안동시", "구미": "구미시", "영주": "영주시", "영천": "영천시", 
        "상주": "상주시", "문경": "문경시", "경산": "경산시", "군위": "군위군", "의성": "의성군", 
        "청송": "청송군", "영양": "영양군", "영덕": "영덕군", "청도": "청도군", "고령": "고령군", 
        "성주": "성주군", "칠곡": "칠곡군", "예천": "예천군", "봉화": "봉화군", "울진": "울진군", 
        "울릉": "울릉군", "창원": "창원시", "진주": "진주시", "통영": "통영시", "사천": "사천시", 
        "김해": "김해시", "밀양": "밀양시", "거제": "거제시", "양산": "양산시", "의령": "의령군", 
        "함안": "함안군", "창녕": "창녕군", "고성": "고성군", "남해": "남해군", "하동": "하동군", 
        "산청": "산청군", "함양": "함양군", "거창": "거창군", "합천": "합천군", "제주": "제주시", 
        "서귀포": "서귀포시"
    }
    
    words = query.split()
    for word in words:
        if word in location_keywords:
            entities["LOCATION"].append(location_keywords[word])
        elif word.endswith(("시", "군", "구")):
            entities["LOCATION"].append(word)
        elif word in major_areas:
            entities["LOCATION"].append(word + major_areas[word])
    
    if "kitchen" in query.lower():
        entities["OCCUPATION"].append("kitchen")

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
            if 'Hwaseong' not in job_info['location']:
                match = False

        # 급여 필터링
        if entities["MONEY"]:
            try:
                # 사용자가 입력한 급여 정보 추출 및 정수로 변환
                required_salary_str = entities["MONEY"][0].replace(',', '').replace('원', '').strip()
                required_salary = int(required_salary_str)
                
                # 직무의 급여 정보 추출 및 정수로 변환
                pay_elements = job_info['pay'].split()
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
            4. **If the query includes a LOCATION keyword, ensure that all relevant job listings for that location are retrieved and included in the response.**  # 추가된 부분
            5. Provide a comprehensive summary of the search results.
            6. Offer detailed information about each relevant job listing.
            7. If the keyword or numerical value does not match the user's query, do not provide any other data.

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