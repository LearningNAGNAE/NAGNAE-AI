from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def saramin_search_crawler(search_queries):
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"  # Update to the actual path

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")  # Set window size

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        all_results = []
        for search_query in search_queries:
            driver.get("https://www.saramin.co.kr/")
            driver.find_element(By.ID, "btn_search").click()
            search_box = driver.find_element(By.ID, "ipt_keyword_recruit")
            search_box.send_keys(search_query)
            driver.find_element(By.ID, "btn_search_recruit").click()

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            search_results = driver.find_elements(By.CLASS_NAME, "item_recruit")

            results = []
            for result in search_results[:5]:
                title_element = result.find_element(By.CSS_SELECTOR, "a.data_layer")
                link_element = title_element.get_attribute("href")
                title = title_element.text
                results.append({"title": title, "link": link_element})

            all_results.extend(results)

        return all_results

    finally:
        driver.quit()

# Example search queries
search_queries = ["외국인 직원", "외국인 가능"]
results = saramin_search_crawler(search_queries)

# Prepare documents for FAISS
documents = [{"text": result['title'], "metadata": {"link": result['link']}} for result in results]

# Initialize OpenAI and FAISS
openai = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings)

# Create retriever tool
retriever = faiss_index.as_retriever()

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are an assistant for foreigners looking for jobs in Korea. Use the following context to help answer the user's questions.
        Context:
        {context}
        Question: {question}
    """
)

# Initialize RetrievalQA with updated method
agent = RetrievalQA(
    llm=openai,
    prompt_template=prompt_template,
    retriever=retriever
)

# Example query to the agent
query = "What are some job opportunities for foreigners in Korea?"
response = agent({"question": query})

print(response["result"])
