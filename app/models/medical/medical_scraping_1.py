from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv

# 환경 변수를 로드합니다.
load_dotenv()

def setup_driver():
    """Chrome WebDriver를 설정하고 반환합니다."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저를 띄우지 않음
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_text_from_page(driver, url, start_xpath, end_class_name):
    """지정된 URL에서 텍스트를 추출합니다."""
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    start_element = wait.until(EC.presence_of_element_located((By.XPATH, start_xpath)))

    elements = []
    capturing = False

    for element in driver.find_elements(By.XPATH, "//*"):
        if capturing:
            elements.append(element)
            if element.tag_name == "p" and end_class_name in element.get_attribute("class"):
                break
        elif element == start_element:
            elements.append(element)
            capturing = True

    extracted_text = "\n".join([element.text for element in elements])
    return extracted_text

def save_text_to_file(text, file_path):
    """텍스트를 파일로 저장합니다."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

def create_faiss_index(text_file_path, faiss_folder_path, chunk_size=300, chunk_overlap=50):
    """FAISS 인덱스를 생성하고 저장합니다."""
    loader = TextLoader(text_file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(faiss_folder_path)

def main():
    driver = setup_driver()
    try:
        url = "https://www.studyinkorea.go.kr/ko/study/KoreaLife03.do"
        start_xpath = "//h3[@class='part' and contains(text(),'국민건강보험')]"
        end_class_name = "source"

        # 현재 파이썬 스크립트의 절대 경로를 얻습니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # "medical_content_1.txt" 파일을 현재 스크립트와 동일한 경로에 저장
        text_file_path = os.path.join(current_dir, "medical_content_1.txt")
        
        # "medical-faiss" 폴더 경로 설정
        medical_faiss_path = os.path.join(current_dir, "medical_faiss")
        
        # 텍스트 추출 및 저장
        extracted_text = extract_text_from_page(driver, url, start_xpath, end_class_name)
        save_text_to_file(extracted_text, text_file_path)

        # FAISS 인덱스 생성 및 저장
        create_faiss_index(text_file_path, medical_faiss_path)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()