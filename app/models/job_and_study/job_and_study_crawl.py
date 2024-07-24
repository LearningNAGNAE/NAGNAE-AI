from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def jobkorea_search_crawler(search_query):
    # ChromeDriver 경로 설정
    chrome_driver_path = r"C:\chromedriver-win64\chromedriver.exe"  # 실제 경로로 변경하세요

    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--headless")  # 헤드리스 모드 추가
    # chrome_options.add_argument("--disable-gpu")  # Windows에서 헤드리스 모드 사용 시 필요할 수 있음
    chrome_options.add_argument("--window-size=1920x1080")  # 창 크기 설정

    # Chrome WebDriver 설정
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 잡코리아 검색 페이지로 이동
        driver.get("https://www.jobkorea.co.kr/")

        # 검색창 찾기 및 검색어 입력
        search_box = driver.find_element(By.NAME, "stext")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)

        # 검색 결과가 로드될 때까지 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.list-default"))
        )

        # 검색 결과 추출
        search_results = driver.find_elements(By.CSS_SELECTOR, "div.list-default div.item")

        results = []
        for result in search_results[:5]:  # 상위 5개 결과만 추출
            title_element = result.find_element(By.CSS_SELECTOR, "a.title")
            link_element = title_element.get_attribute("href")

            title = title_element.text
            
            results.append({"title": title, "link": link_element})

        return results

    finally:
        # 브라우저 종료
        driver.quit()

# 크롤러 실행
search_query = "Python 프로그래밍"
results = jobkorea_search_crawler(search_query)

# 결과 출력
for idx, result in enumerate(results, 1):
    print(f"{idx}. Title: {result['title']}")
    print(f"   Link: {result['link']}")
    print()
