from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json

def jobploy_crawler(lang, pages=1):
    if isinstance(pages, dict):
        pages = 1
    elif not isinstance(pages, int):
        try:
            pages = int(pages)
        except ValueError:
            pages = 1

    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")
    # chrome_options.add_argument("--headless")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []

    try:
        for page in range(1, pages + 1):
            url = f"https://www.jobploy.kr/{lang}/recruit?page={page}"
            driver.get(url)

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

            for job in job_listings:
                title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
                link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                pay_element = job.find_element(By.CSS_SELECTOR, "p.pay")
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")

                if len(badge_elements) >= 3:
                    location_element = badge_elements[0]
                    task_element = badge_elements[1]
                    closing_date_element = badge_elements[2]
                    location = location_element.text
                    task = task_element.text
                    closing_date = closing_date_element.text
                else:
                    closing_date = "마감 정보 없음"
                    location = "위치 정보 없음"
                    task = "직무 정보 없음"

                title = title_element.text
                pay = pay_element.text
                
                results.append({
                    "title": title,
                    "link": link_element,
                    "closing_date": closing_date,
                    "location": location,
                    "pay": pay,
                    "task": task,
                    "language": lang
                })

    finally:
        driver.quit()
        
    return results

# # 언어별 데이터 저장 함수
# def save_data_to_file(data, filename):
#     with open(filename, "w", encoding="utf-8") as file:
#         json.dump(data, file, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
    # results = jobploy_crawler('ko', pages=1)
    