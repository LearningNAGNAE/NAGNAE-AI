from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

load_dotenv()

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
                
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")
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

chat_openai = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
embeddings = OpenAIEmbeddings()

def prepare_documents():
    results = jobploy_crawler()
    documents = [{"text": result['title'], "metadata": {"link": result['link']}} for result in results]
    faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings, metadatas=[doc["metadata"] for doc in documents])
    return faiss_index

faiss_index = prepare_documents()
retriever = faiss_index.as_retriever()

agent = ConversationalRetrievalChain.from_llm(
    llm=chat_openai,
    retriever=retriever,
    return_source_documents=True
)

def get_agent_response(query):
    inputs = {"question": query, "context": "", "chat_history": []}
    response = agent.invoke(inputs)
    return response