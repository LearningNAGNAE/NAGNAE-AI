import requests
from bs4 import BeautifulSoup, NavigableString, Comment
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_community.document_loaders import TextLoader
import os
import re
import time

# 현재 스크립트의 디렉토리 경로를 얻습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

def setup_driver():
    """Chrome WebDriver를 설정하고 반환합니다."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저를 띄우지 않음
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_text_from_page_1(driver, url, start_xpath, end_class_name):
    """지정된 URL에서 특정 시작 요소와 끝 요소 사이의 텍스트를 추출합니다."""
    driver.get(url)

    try:
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
    except Exception as e:
        print(f"Error in extract_text_from_page_1: {e}")
        return ""

def save_text_to_file_1(text, file_path):
    """텍스트를 지정된 파일 경로에 저장합니다."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text saved to {file_path}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

def process_text(file_path):
    """지정된 파일 경로의 텍스트 파일을 로드하고 텍스트를 추출합니다."""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print('process_text 메소드 완료')
        return documents
    except Exception as e:
        print(f"Error processing text file: {e}")
        return []

def fetch_page(url):
    """지정된 URL의 HTML 콘텐츠를 가져옵니다."""
    try:
        response = requests.get(url)
        response.encoding = 'utf-8'
        print(f"Status code: {response.status_code}")  # 상태 코드 출력
        return response.text
    except Exception as e:
        print(f"Error fetching page: {e}")
        return ""

def parse_html_1(html):
    """HTML 문자열을 파싱하여 BeautifulSoup 객체를 반환합니다."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        return soup
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None

def extract_text_from_page_2(soup, start_class, end_class):
    """BeautifulSoup 객체에서 특정 시작 클래스와 끝 클래스 사이의 텍스트를 추출합니다."""
    if soup is None:
        print("Soup object is None")
        return []

    start_tag = soup.find('h5', class_=start_class)
    if start_tag is None:
        print(f"Start tag with class '{start_class}' not found")
        return []

    end_tag = soup.find('div', class_=end_class)
    if end_tag is None:
        print(f"End tag with class '{end_class}' not found")
    
    content = []
    current_tag = start_tag
    
    while current_tag and current_tag != end_tag:
        content.append(current_tag.get_text(strip=True))
        current_tag = current_tag.find_next_sibling()
    
    return content

def save_text_to_file_2(content, filename):
    """텍스트 리스트를 지정된 파일 경로에 저장합니다."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for line in content:
                file.write(line + "\n")
        print(f"Text saved to {filename}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

def save_text_to_file_3(content, filename):
    """텍스트 리스트를 지정된 파일 경로에 저장합니다."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Text saved to {filename}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

def scrape_content(driver, url):
    try:
        driver.get(url)
        
        iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "lawService")))
        driver.switch_to.frame(iframe)
        time.sleep(5)
        
        html_content = driver.page_source
        return html_content
    except Exception as e:
        print(f"Error scraping content: {e}")
        return ""

def parse_html_2(html_content):
    try:
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
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None
    
def extract_text_from_page_4(soup, tag, selected_class):
    content = soup.find_all(tag, class_= selected_class)
    return content

def save_text_to_file_3(contents, filename):
    """텍스트 리스트를 지정된 파일 경로에 저장합니다."""
    with open(filename, 'w', encoding='utf-8') as file:
        for div in contents:
            text_content = div.get_text(strip=True, separator='\n')
            file.write(text_content + "\n\n")

def main():
    """메인 함수: 웹 페이지에서 텍스트를 추출하여 파일로 저장하고, 텍스트 파일을 로드하여 문서로 반환합니다."""
    driver = setup_driver()

    try:
        # 첫번째 URL 텍스트 추출
        first_url = "https://www.studyinkorea.go.kr/ko/study/KoreaLife03.do"
        start_xpath = "//h3[@class='part' and contains(text(),'국민건강보험')]"
        end_class_name = "source"
        first_file_path = os.path.join(current_dir, "scraping_file", "national_health_insurance.txt")
        extracted_text = extract_text_from_page_1(driver, first_url, start_xpath, end_class_name)
        save_text_to_file_1(extracted_text, first_file_path)

        # 두번째 URL 텍스트 추출
        second_url = "https://www.hira.or.kr/dummy.do?pgmid=HIRAA020020000003"
        second_html = fetch_page(second_url)
        second_soup = parse_html_1(second_html)
        second_content = extract_text_from_page_2(second_soup, 'tit_square01', 'imgBox mt30')
        second_file_path = os.path.join(current_dir, "scraping_file", "medical_expenses.txt")
        save_text_to_file_2(second_content, second_file_path)

        # 세번째 URL 텍스트 추출
        third_url = "https://www.law.go.kr/%ED%96%89%EC%A0%95%EA%B7%9C%EC%B9%99/%EC%9E%A5%EA%B8%B0%EC%B2%B4%EB%A5%98%20%EC%9E%AC%EC%99%B8%EA%B5%AD%EB%AF%BC%20%EB%B0%8F%20%EC%99%B8%EA%B5%AD%EC%9D%B8%EC%97%90%20%EB%8C%80%ED%95%9C%20%EA%B1%B4%EA%B0%95%EB%B3%B4%ED%97%98%20%EC%A0%81%EC%9A%A9%EA%B8%B0%EC%A4%80"
        third_file_path = os.path.join(current_dir, "scraping_file", "long_term stay_health_insurance_application_criteria.txt")
        third_content = scrape_content(driver, third_url)
        parsed_content = parse_html_2(third_content)
        save_text_to_file_1(parsed_content, third_file_path)

        # 네번째 URL 텍스트 추출
        fourth_url = "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=508&ccfNo=3&cciNo=6&cnpClsNo=2&menuType=cnpcls&search_put="
        fourth_html = fetch_page(fourth_url)
        fourth_soup = parse_html_1(fourth_html)
        fourth_content = extract_text_from_page_4(fourth_soup, 'div', 'ovDivbox')
        fourth_file_path = os.path.join(current_dir, "scraping_file", "sign_up_for_health_insurance.txt")
        save_text_to_file_3(fourth_content, fourth_file_path)

        # 다섯번째 URL 텍스트 추출
        fifth_url = "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=508&ccfNo=3&cciNo=6&cnpClsNo=2&menuType=onhunqna&search_put="
        fifth_html = fetch_page(fifth_url)
        fifth_soup = parse_html_1(fifth_html)
        fifth_content = extract_text_from_page_4(fifth_soup, 'dl', 'q')
        fifth_file_path = os.path.join(current_dir, "scraping_file", "sign_up_for_health_insurance_question.txt")
        save_text_to_file_3(fifth_content, fifth_file_path)

        # 여섯번째 URL 텍스트 추출
        sixth_url = "https://www.easylaw.go.kr/CSP/CnpClsMain.laf?popMenu=ov&csmSeq=508&ccfNo=3&cciNo=6&cnpClsNo=1&search_put="
        sixth_html = fetch_page(sixth_url)
        sixth_soup = parse_html_1(sixth_html)
        sixth_content = extract_text_from_page_4(sixth_soup, 'div', 'ovDivbox')
        sixth_file_path = os.path.join(current_dir, "scraping_file", "medical_system.txt")
        save_text_to_file_3(sixth_content, sixth_file_path)

        # 합치기
        first_documents = process_text(first_file_path)
        second_documents = process_text(second_file_path)
        third_documents = process_text(third_file_path)
        fourth_documents = process_text(fourth_file_path)
        fifth_documents = process_text(fifth_file_path)
        sixth_documents = process_text(sixth_file_path)

        documents = first_documents + second_documents + third_documents + fourth_documents + fifth_documents + sixth_documents
        
        return documents

    except Exception as e:
        print(f"Error in main function: {e}")
        return []

    finally:
        driver.quit()

if __name__ == "__main__":
    main()