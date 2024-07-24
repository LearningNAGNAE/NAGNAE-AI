from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 구인구직 사이트 크롤러 설정
def job_site_crawler():
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"  # 실제 경로로 변경하세요
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get("https://www.example-job-site.com/")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input.search-box"))
        )
        search_box = driver.find_element(By.CSS_SELECTOR, "input.search-box")
        search_box.send_keys("software engineer")
        search_box.send_keys(Keys.RETURN)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.job-listing"))
        )

        job_elements = driver.find_elements(By.CSS_SELECTOR, "div.job-listing")
        jobs = []

        for job in job_elements:
            title = job.find_element(By.CSS_SELECTOR, "h2.title").text
            company = job.find_element(By.CSS_SELECTOR, "div.company").text
            link = job.find_element(By.CSS_SELECTOR, "a.apply-link").get_attribute("href")
            jobs.append({"title": title, "company": company, "link": link})

        return jobs

    finally:
        driver.quit()

# 주기적으로 데이터를 갱신
def update_job_data():
    jobs = job_site_crawler()
    documents = [{"content": job["title"] + " " + job["company"], "metadata": job} for job in jobs]
    chunked_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    vectordb = FAISS.from_documents(chunked_docs, OpenAIEmbeddings())
    return vectordb.as_retriever()

# 최초 데이터 로딩
retriever = update_job_data()

# 에이전트 생성
prompt = hub.pull("hwchase17/openai-functions-agent")
openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)

retriever_tool = create_retriever_tool(
    retriever, "job_search", "구인구직 정보를 검색하세요!"
)

tools = [retriever_tool]
agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 에이전트 실행
agent_result = agent_executor.invoke({"input": "소프트웨어 엔지니어 채용 공고를 알려줘"})
print(agent_result)

# 주기적인 데이터 갱신을 위한 스케줄링 (예: 하루에 한 번)
def schedule_update():
    while True:
        time.sleep(86400)  # 86400초 = 24시간
        global retriever
        retriever = update_job_data()

# 스케줄러 실행 (백그라운드에서 실행되도록 설정)
import threading
threading.Thread(target=schedule_update, daemon=True).start()
