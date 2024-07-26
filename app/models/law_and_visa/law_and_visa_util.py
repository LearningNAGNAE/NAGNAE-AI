from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # ChromeDriver 경로를 직접 지정
    chromedriver_path = r"C:\\chromedriver\\chromedriver.exe"  # 실제 경로로 변경하세요
    service = Service(chromedriver_path)

    return webdriver.Chrome(service=service, options=chrome_options)

def wait_for_element(driver, by, value, timeout=30):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )

def wait_for_element_safely(driver, by, value, timeout=30):
    try:
        return wait_for_element(driver, by, value, timeout)
    except TimeoutException:
        print(f"요소를 찾을 수 없습니다: {by}={value}")
        return None