from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import os
import time
import PyPDF2

def download_pdf(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def google_search_crawler(search_query):
    # ChromeDriver 경로 설정
    chrome_driver_path = r"C:\chromedriver-win64\chromedriver.exe"  # 실제 경로로 변경하세요

    # 다운로드 경로 설정
    download_folder = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Chrome 옵션 설정
    chrome_options = Options()
    prefs = {"download.default_directory": download_folder}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")  # 헤드리스 모드 추가
    chrome_options.add_argument("--disable-gpu")  # Windows에서 헤드리스 모드 사용 시 필요할 수 있음
    chrome_options.add_argument("--window-size=1920x1080")  # 창 크기 설정

    # Chrome WebDriver 설정
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 검색 페이지로 이동
        driver.get("https://www.adiga.kr/man/inf/mainView.do?menuId=PCMANINF1000")

        # 검색창 찾기 및 검색어 입력
        search_box = driver.find_element(By.CLASS_NAME, "XSSSafeInput")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        # 검색 결과가 로드될 때까지 대기
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "boardCon"))
        )

        # 검색 결과 클릭
        search_results = driver.find_elements(By.CSS_SELECTOR, "ul.uctList01 li")

        results = []
        for result in search_results[:3]:  # 상위 3개 결과만 추출
            title_element = result.find_element(By.CSS_SELECTOR, "p")
            link_element = result.find_element(By.CSS_SELECTOR, "a")
            
            title = title_element.text
            link = link_element.get_attribute("href")
            
            results.append({"title": title, "link": link})

        # 첫 번째 검색 결과 페이지로 이동
        driver.get(results[0]["link"])

        # 검색 결과가 로드될 때까지 대기
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "popCont"))
        )

        # fnFileDownOne 함수를 호출하는 첫 번째 <a> 태그 찾기
        download_link = driver.find_element(By.XPATH, "//a[contains(@onclick, 'fnFileDownOne')]")
        onclick_text = download_link.get_attribute("onclick")

        # JavaScript 함수를 직접 실행하여 PDF 다운로드
        driver.execute_script(onclick_text)

        # 다운로드된 파일의 경로 확인
        files_before = set(os.listdir(download_folder))
        # print(f"Files before download: {files_before}")  # 디버깅 출력
        time.sleep(10)  # 다운로드 대기
        files_after = set(os.listdir(download_folder))
        # print(f"Files after download: {files_after}")  # 디버깅 출력

        new_files = files_after - files_before
        # print(f"New files: {new_files}")  # 디버깅 출력

        if not new_files:
            raise FileNotFoundError("PDF 파일을 찾을 수 없습니다. 다운로드 폴더: {}".format(download_folder))

        # 새로 다운로드된 파일의 이름 확인
        downloaded_file_name = new_files.pop()
        if not downloaded_file_name.endswith(".pdf"):
            raise FileNotFoundError(f"PDF 파일이 아닙니다: {downloaded_file_name}")

        pdf_path = os.path.join(download_folder, downloaded_file_name)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # PDF 파일에서 텍스트 추출
        pdf_text = extract_text_from_pdf(pdf_path)
        print(pdf_text)

        return pdf_text
    finally:
        # 브라우저 종료
        driver.quit()

# 크롤러 실행
search_query = "외국인 특별전형 시행계획 주요사항"
results = google_search_crawler(search_query)
