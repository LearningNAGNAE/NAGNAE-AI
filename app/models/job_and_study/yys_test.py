from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from selenium import webdriver
from langchain_core.documents import Document
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# 글로벌 변수로 agent_executor 선언
agent_executor = None

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
                pay_element = job.find_element(By.CSS_SELECTOR, "p.pay")
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")

                if len(badge_elements) >= 3:
                    location_element = badge_elements[0]
                    closing_date_element = badge_elements[2]
                    location = location_element.text
                    closing_date = closing_date_element.text
                else:
                    closing_date = "마감 정보 없음"
                    location = "위치 정보 없음"

                title = title_element.text
                pay = pay_element.text
                results.append({"title": title, "link": link_element, "closing_date": closing_date, "location": location, "pay": pay})

                print(results)
    finally:
        driver.quit()

    return results

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행될 코드
    global agent_executor
    
    # 크롤링 데이터 가져오기
    crawled_data = jobploy_crawler(pages=5)

    # 크롤링한 데이터를 Document 객체로 변환
    documents = [
        Document(
            page_content=f"Title: {job['title']}\nLink: {job['link']}\nClosing Date: {job['closing_date']}\nLocation: {job['location']}\nPay: {job['pay']}",
            metadata={"source": job['link']}
        ) for job in crawled_data
    ]

    # 벡터 데이터베이스 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 검색기 생성
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 검색기 도구 생성
    retriever_tool = create_retriever_tool(
        retriever,
        "job_search",
        "Use this tool to search for job information from Jobploy website."
    )

    # LLM 모델 설정
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Agent 프롬프트 설정
    prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt = prompt.partial(
        system_message="You are an AI assistant specializing in job search. Your task is to provide accurate and relevant information about job listings from the Jobploy website. When answering questions, use the information from the job listings directly, and avoid making assumptions or providing information not present in the data."
    )

    # Agent 생성
    agent = create_openai_tools_agent(llm=llm, tools=[retriever_tool], prompt=prompt)

    # Agent Executor 생성
    agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

    yield

    # 종료 시 실행될 코드
    # 필요한 경우 여기에 정리 코드를 추가할 수 있습니다.

app = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    input: str

@app.post("/job_search")
async def job_search(query: Query):
    global agent_executor
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="Agent executor not initialized")
    
    try:
        result = agent_executor.invoke({"input": query.input})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)