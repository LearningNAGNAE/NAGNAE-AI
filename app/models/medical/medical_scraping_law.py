import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, NavigableString, Comment
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def setup_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def scrape_content(url):
    driver = setup_chrome_driver()
    driver.get(url)
    
    iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "lawService")))
    driver.switch_to.frame(iframe)
    time.sleep(5)
    
    html_content = driver.page_source
    driver.quit()
    return html_content

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    con_top = soup.find('div', id='conTop')
    con_scroll = soup.find('div', id='conScroll')
    
    if con_top and con_scroll:
        content = []
        current_paragraph = ""
        
        def process_element(element):
            nonlocal current_paragraph
            if isinstance(element, Comment):
                return  # Skip comments
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    current_paragraph += text + " "
            elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_paragraph:
                    content.append(current_paragraph.strip())
                    current_paragraph = ""
                content.append(element.get_text(strip=True))
            elif element.name == 'p':
                current_paragraph += element.get_text(strip=True) + " "
            elif element.name == 'br':
                if current_paragraph:
                    content.append(current_paragraph.strip())
                    current_paragraph = ""
            else:
                for child in element.children:
                    process_element(child)
        
        current = con_top
        while current and current != con_scroll:
            process_element(current)
            current = current.next_sibling
        
        process_element(con_scroll)
        
        if current_paragraph:
            content.append(current_paragraph.strip())
        
        # Join all content without line breaks
        final_content = " ".join(content)
        
        # Remove any remaining HTML comments
        final_content = re.sub(r'<!--.*?-->', '', final_content, flags=re.DOTALL)
        
        return final_content.strip()
    return None

def save_content(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("내용이 성공적으로 저장되었습니다.")

def create_faiss_vector_db(text, embeddings, db_path):
    db = FAISS.from_texts([text], embeddings)
    db.save_local(db_path)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "scraping_file", "long_term stay_health_insurance_application_criteria.txt")
    medical_faiss_path = os.path.join(current_dir, "medical_faiss")
    
    url = "https://www.law.go.kr/%ED%96%89%EC%A0%95%EA%B7%9C%EC%B9%99/%EC%9E%A5%EA%B8%B0%EC%B2%B4%EB%A5%98%20%EC%9E%AC%EC%99%B8%EA%B5%AD%EB%AF%BC%20%EB%B0%8F%20%EC%99%B8%EA%B5%AD%EC%9D%B8%EC%97%90%20%EB%8C%80%ED%95%9C%20%EA%B1%B4%EA%B0%95%EB%B3%B4%ED%97%98%20%EC%A0%81%EC%9A%A9%EA%B8%B0%EC%A4%80"
    
    html_content = scrape_content(url)
    parsed_content = parse_html(html_content)
    
    if parsed_content:
        save_content(parsed_content, file_path)
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        create_faiss_vector_db(parsed_content, embeddings, medical_faiss_path)
        print("벡터 데이터베이스가 성공적으로 생성되었습니다.")
    else:
        print("원하는 태그를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()