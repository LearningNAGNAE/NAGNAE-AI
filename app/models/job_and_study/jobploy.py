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
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain import hub
import os

# Load environment variables
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

# Test the crawler function
results = jobploy_crawler()
print(results)   

# Prepare documents for FAISS
documents = [{"text": result['title'], "metadata": {"link": result['link']}} for result in results]

# pip install langchainhub
# hub에서 가져온 prompt를 agent에게 전달하기 위한 prompt 생성   
prompt = hub.pull("hwchase17/openai-functions-agent")

# Initialize OpenAI and FAISS
chat_openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature= 0.1)
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings, metadatas=[doc["metadata"] for doc in documents])

# Create retriever tool
retriever = faiss_index.as_retriever()

# # Define prompt template
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#         You are an assistant for foreigners looking for jobs in Korea. Use the following context to help answer the user's questions.
#         Context:
#         {context}
#         Question: {question}
#     """
# )

# Define prompt template
# prompt_template = hub.pull("hwchase17/openai-functions-agent")

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
    # 응답에서 메시지를 가져옵니다.
    generated_question = response.content.strip()

    return generated_question


# Example query to the agent
query = ""  # 빈 질문으로 설정하여 에이전트가 자체적으로 질문 생성

# 빈 질문인 경우 질문을 생성합니다.
if not query:
    query = generate_question_based_on_documents(documents)

# 생성된 질문 출력
print("생성된 질문:", query)

inputs = {"question": query, "context": "", "chat_history": []}  # Include chat_history
response = agent.invoke(inputs)

# 응답에 링크 추가
response_with_links = response.copy()
response_with_links["answer"] = response["answer"]
for doc in response["source_documents"]:
    title = doc.page_content
    link = doc.metadata.get("link", "https://www.jobploy.kr/ko/recruit")  # 링크가 없는 경우를 대비해 기본값 설정
    response_with_links["answer"] += f"\n- {title}: {link}"

print(response_with_links)
