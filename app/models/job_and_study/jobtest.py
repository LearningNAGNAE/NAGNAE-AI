from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

app = FastAPI()

# CORS 설정
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def jobploy_crawler(pages=5):
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"  # Update to the actual path

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")  # Set window size

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []
    try:
        for page in range(1, pages+1):
            driver.get(f"https://www.jobploy.kr/ko/recruit?page={page}")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

            for job in job_listings:
                title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
                link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                location_element = job.find_element(By.CSS_SELECTOR, ".item_link")
                salary_element = job.find_element(By.CSS_SELECTOR, ".pay")
                
                # 모든 배지 요소를 가져옴
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")
                # 첫 번째 배지 요소(지역 정보) 선택
                if len(badge_elements) >= 1:
                    location_element = badge_elements[0]
                    location = location_element.text
                else:
                    location = "지역 정보 없음"
                
                # 세 번째 배지 요소(마감 정보) 선택
                if len(badge_elements) >= 3:
                    closing_date_element = badge_elements[2]
                    closing_date = closing_date_element.text
                else:
                    closing_date = "마감 정보 없음"

                title = title_element.text
                salary = salary_element.text

                results.append({
                    "title": title,
                    "link": link_element,
                    "location": location,
                    "salary": salary,
                    "closing_date": closing_date
                })

    finally:
        driver.quit()
    
    return results

# Initialize OpenAI and FAISS
chat_openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
embeddings = OpenAIEmbeddings()

def prepare_documents():
    results = jobploy_crawler()
    documents = []

    for idx, result in enumerate(results):
        # 고유 식별자 생성 (예: 공고 링크 또는 인덱스 사용)
        unique_id = result['link'] if 'link' in result and result['link'] else str(idx)
        
        # 중복 가능성을 줄이기 위해 공고의 주요 정보를 포함한 텍스트 생성
        full_text = f"ID: {unique_id}\nTitle: {result['title']}\n Location: {result['location']}\n Salary: {result['salary']}\n Closing Date: {result['closing_date']}\n Link: {result['link']}"
        metadata = {
            "link": result['link'],
            "location": result['location'],
            "salary": result['salary'],
            "closing_date": result['closing_date'],
            "unique_id": unique_id
        }
        documents.append({"text": full_text, "metadata": metadata})

    # 모든 텍스트를 FAISS 인덱스에 저장
    faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings, metadatas=[doc["metadata"] for doc in documents])
    return faiss_index

# Prepare the FAISS index
faiss_index = prepare_documents()

# Create retriever tool
retriever = faiss_index.as_retriever(search_type="similarity")

# Load the prompt from the hub
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt = prompt.partial(
    system_message="You are an AI assistant specializing in job search. Your task is to provide accurate and relevant information about job listings from the Jobploy website. When answering questions, use the information from the job listings directly, and avoid making assumptions or providing information not present in the data."
)

# Create the retriever tool
retriever_tool = create_retriever_tool(
    retriever, 
    "job_search", 
    "Use this tool to search for job information from Jobploy website."
)

# Create agent with the prompt and tool
agent = create_openai_tools_agent(llm=chat_openai, tools=[retriever_tool], prompt=prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the JobPloy API"}

@app.post("/ask")
async def ask(query_request: QueryRequest):
    user_query = query_request.query

    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required")

    global agent_executor
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="Agent executor not initialized")
    
    try:
        # `agent_executor`를 사용하여 요청 처리
        result = agent_executor.invoke({"input": user_query})
        
        # 반환된 결과를 적절히 처리
        response_with_links = result.copy()
        response_with_links["answer"] = "Here are the jobs that match your query:<br>"
        
        # 문서가 없을 경우를 대비해 빈 리스트를 기본값으로 설정
        source_documents = result.get("source_documents", [])
        
        # 응답 문자열을 문서 정보로 작성
        for doc in source_documents:
            title = doc.page_content
            link = doc.metadata.get("link", "https://www.jobploy.kr/ko/recruit")
            location = doc.metadata.get("location", "지역 정보 없음")
            salary = doc.metadata.get("salary", "급여 정보 없음")
            closing_date = doc.metadata.get("closing_date", "마감 정보 없음")
            response_with_links["answer"] += f"<br>- {title} (위치: {location}, 급여: {salary}, 마감일: {closing_date}): {link}"

        return response_with_links

    except Exception as e:
        # 예외 발생 시 로그
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
