from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain import hub
import os
from fastapi.middleware.cors import CORSMiddleware

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
                
                # 모든 배지 요소를 가져옴
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")
                # 세 번째 배지 요소(마감 정보) 선택
                if len(badge_elements) >= 3:
                    closing_date_element = badge_elements[2]
                    closing_date = closing_date_element.text
                else:
                    closing_date = "마감 정보 없음"

                title = title_element.text
                results.append({"title": title, "link": link_element, "closing_date": closing_date})

    finally:
        driver.quit()
    
    return results

# Initialize OpenAI and FAISS
chat_openai = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
embeddings = OpenAIEmbeddings()

def prepare_documents():
    results = jobploy_crawler()
    documents = [{"text": result['title'], "metadata": {"link": result['link']}} for result in results]
    faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings, metadatas=[doc["metadata"] for doc in documents])
    return faiss_index

faiss_index = prepare_documents()

# Create retriever tool
retriever = faiss_index.as_retriever()

# Initialize ConversationalRetrievalChain
agent = ConversationalRetrievalChain.from_llm(
    llm=chat_openai,
    retriever=retriever,
    return_source_documents=True
)

@app.post("/ask")
async def ask(query_request: QueryRequest):
    user_query = query_request.query

    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required")

    inputs = {"question": user_query, "context": "", "chat_history": []}  # Include chat_history
    response = agent.invoke(inputs)

    # 응답에 링크 추가
    response_with_links = response.copy()
    response_with_links["answer"] = response["answer"]
    for doc in response["source_documents"]:
        title = doc.page_content
        link = doc.metadata.get("link", "https://www.jobploy.kr/ko/recruit")  # 링크가 없는 경우를 대비해 기본값 설정
        response_with_links["answer"] += f"\n- {title}: {link}"

    return response_with_links

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
