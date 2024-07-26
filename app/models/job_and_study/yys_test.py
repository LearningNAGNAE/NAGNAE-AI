from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import time

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행할 코드
    setup_langchain()
    yield
    # 종료 시 실행할 코드 (필요한 경우)

app = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    input: str

def study_search_crawler(search_query: str) -> str:
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"  # Update this path as needed
    download_folder = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    chrome_options = Options()
    prefs = {"download.default_directory": download_folder}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get("https://www.adiga.kr/man/inf/mainView.do?menuId=PCMANINF1000")
        search_box = driver.find_element(By.CLASS_NAME, "XSSSafeInput")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "boardCon"))
        )

        search_results = driver.find_elements(By.CSS_SELECTOR, "ul.uctList01 li")

        results = []
        for result in search_results[:3]:
            title_element = result.find_element(By.CSS_SELECTOR, "p")
            link_element = result.find_element(By.CSS_SELECTOR, "a")
            title = title_element.text
            link = link_element.get_attribute("href")
            results.append({"title": title, "link": link})

        driver.get(results[0]["link"])
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "popCont"))
        )

        download_link = driver.find_element(By.XPATH, "//a[contains(@onclick, 'fnFileDownOne')]")
        onclick_text = download_link.get_attribute("onclick")

        # 기존 PDF 파일 삭제
        for file_name in os.listdir(download_folder):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(download_folder, file_name)
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")

        # 파일 다운로드
        driver.execute_script(onclick_text)

        # 다운로드 완료 대기
        time.sleep(10)

        # 다운로드된 파일 찾기
        files_after = set(os.listdir(download_folder))
        downloaded_files = [f for f in files_after if f.endswith(".pdf")]

        if not downloaded_files:
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다. 다운로드 폴더: {download_folder}")

        pdf_path = os.path.join(download_folder, downloaded_files[0])

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        return pdf_path
    finally:
        driver.quit()

def setup_langchain():
    global agent_executor
    search_query = "외국인 특별전형 시행계획 주요사항"
    pdf_path = study_search_crawler(search_query)
    print(f"Downloaded PDF file: {pdf_path}")

    # Langchain setup
    prompt = hub.pull("hwchase17/openai-functions-agent")

    openai = ChatOpenAI(
        model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)

    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split documents into chunks and create vector database
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()

    print(retriever)

    retriever_tool = create_retriever_tool(
        retriever, "pdf_search", "PDF 파일에서 추출한 정보를 검색할 때 이 툴을 사용하세요.")
    print(retriever_tool.name)

    # Define tools for the agent
    tools = [retriever_tool]

    # Define the agent and create an executor
    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/query")
def query_agent(query: Query):
    try:
        agent_result = agent_executor.invoke({"input": query.input})
        return {"result": agent_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)