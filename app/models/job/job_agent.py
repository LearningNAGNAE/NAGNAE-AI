from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, LLMResult
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
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
chat_openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature= 0.7)
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

# Initialize ConversationalRetrievalChain
agent = ConversationalRetrievalChain.from_llm(
    llm=chat_openai,
    retriever=retriever,
    return_source_documents = True
)
def generate_question_based_on_documents(documents):
    # 이 함수는 임의의 질문을 생성하는 데 사용됩니다.
    # 임베딩된 문서 기반으로 질문을 생성합니다.
    if not documents:
        return "What job opportunities are available for foreigners in Korea?"

    context = "\n".join([doc["text"] for doc in documents])
    prompt = f"Generate a question based on the following context:\n{context}"
    response = chat_openai(messages=[HumanMessage(content=prompt)])  # messages 매개변수로 HumanMessage 객체의 리스트를 전달합니다.
    generated_question = response["choices"][0]["message"]["content"].strip()

    return generated_question

# Example query to the agent
query = ""  # 빈 질문으로 설정하여 에이전트가 자체적으로 질문 생성

# 빈 질문인 경우 질문을 생성합니다.
if not query:
    query = generate_question_based_on_documents(documents)

inputs = {"question": query, "context": "", "chat_history": []}  # Include chat_history
response = agent.invoke(inputs)

# 응답에 링크 추가
response_with_links = response.copy()
for doc in response["source_documents"]:
    title = doc.page_content
    link = doc.metadata["link"]
    response_with_links["answer"] += f"\n- {title}: {link}"

print(response_with_links)