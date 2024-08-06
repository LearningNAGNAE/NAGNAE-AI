from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain_community.document_loaders import PyPDFLoader
import os, time
# import logging
from dotenv import load_dotenv

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

# 로깅 설정
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

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
    chrome_options.add_argument("--headless")  # 이 줄을 추가합니다

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

async def setup_langchain():
    search_query = "외국인 특별전형 시행계획 주요사항"
    pdf_path = study_search_crawler(search_query)
    # logger.info(f"Downloaded PDF file: {pdf_path}")

    system_message = SystemMessage(content="""
    You are an AI assistant specialized in analyzing and answering questions about university admission policies for international students in Korea. 
    Your primary task is to provide accurate and helpful information based on the PDF documents you have access to.
    When answering questions:
    1. Always refer to the information in the PDF documents.
    2. If you're not sure about something, say so rather than guessing.
    3. Provide specific details and cite the relevant sections of the document when possible.
    4. If a question is outside the scope of the information in the documents, politely inform the user.
    5. Be concise in your responses, but provide enough detail to be helpful.
    6. Respond in the same language as the input question. If the input is in English, respond in English. If it's in Korean, respond in Korean. For other languages, respond in English.
    """)

    human_message = HumanMessage(content="{input}")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Language: {language}\n\nQuestion: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    openai = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)

    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Set up retrievers
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    async def get_context(question):
        docs = await ensemble_retriever.aget_relevant_documents(question)
        return "\n".join(doc.page_content for doc in docs)

    retrieval_chain = (
        {
            "context": lambda x: get_context(x["question"]),
            "question": lambda x: x["question"],
            "language": lambda x: detect_language(x["question"])
        }
        | chat_prompt
        | openai
        | StrOutputParser()
    )

    async def detect_language(input):
        system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
        human_prompt = f"Text: {input}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ]
        response = await openai.ainvoke(messages)
        detected_language = response.content.strip().lower()
        # logger.debug(f"Detected language: {detected_language}")
        return detected_language

    # 비동기 함수로 변경
    async def run_chain(input_text):
        language = await detect_language(input_text)
        context = await get_context(input_text)
        response = await retrieval_chain.ainvoke({
            "question": input_text,
            "language": language,
            "context": context
        })
        return f"Detected language: {language}\n\nResponse: {response}"
    
    # Create retriever tool
    retriever_tool = create_retriever_tool(
        ensemble_retriever, "pdf_search", "PDF 파일에서 추출한 정보를 검색할 때 이 툴을 사용하세요.")

    # Define tools for the agent
    tools = [retriever_tool, Tool(name="run_chain", func=run_chain, description="Use this to process the input through the retrieval chain")]

    # Define the agent and create an executor
    agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=chat_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    return agent_executor