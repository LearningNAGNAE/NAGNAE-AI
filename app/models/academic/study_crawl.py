import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def study_search_crawler():
    search_query = "외국인 특별전형 시행계획 주요사항"

    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"  # Update this path as needed
    download_folder = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    chrome_options = Options()
    prefs = {"download.default_directory": download_folder}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--headless")  # Run headless mode

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get("https://www.adiga.kr/man/inf/mainView.do?menuId=PCMANINF1000")
        search_box = driver.find_element(By.CLASS_NAME, "XSSSafeInput")
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "boardCon"))
        )

        search_results = driver.find_elements(By.CSS_SELECTOR, "ul.uctList01 li")

        results = []
        for result in search_results[:3]:
            title_element = result.find_element(By.CSS_SELECTOR, "p")
            link_element = result.find_element(By.CSS_SELECTOR, "a")
            title = title_element.text
            link = link_element.get_attribute("href")
            results.append({"title": title, "link": link})

        driver.get(results[0]["link"])
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "popCont"))
        )

        download_link = driver.find_element(By.XPATH, "//a[contains(@onclick, 'fnFileDownOne')]")
        onclick_text = download_link.get_attribute("onclick")

        # Remove existing PDF files
        for file_name in os.listdir(download_folder):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(download_folder, file_name)
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")

        # Download the file
        driver.execute_script(onclick_text)

        # Wait for the download to complete
        time.sleep(10)

        # Find the downloaded file
        files_after = set(os.listdir(download_folder))
        downloaded_files = [f for f in files_after if f.endswith(".pdf")]

        if not downloaded_files:
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다. 다운로드 폴더: {download_folder}")

        pdf_path = os.path.join(download_folder, downloaded_files[0])

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # Rename the downloaded file
        new_pdf_name = "외국인 전형 대학 정보.pdf"  # Change this to your desired filename
        new_pdf_path = os.path.join(download_folder, new_pdf_name)

        os.rename(pdf_path, new_pdf_path)
        print(f"Renamed file to: {new_pdf_path}")

        return pdf_path
    finally:
        driver.quit()

# Example usage
# search_query = "외국인 특별전형 시행계획 주요사항"
pdf_path = study_search_crawler()
# print(f"Downloaded PDF file: {pdf_path}")
