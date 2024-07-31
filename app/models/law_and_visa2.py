from httpcore import TimeoutException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_content(driver, is_main=False):
    try:
        if is_main:
            content = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "scr_ctrl"))
            )
        else:
            content = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        return content.text
    except TimeoutException:
        print(f"내용을 찾을 수 없습니다.")
        return ""

def scrape_law_website():
    base_url = "https://www.law.go.kr"
    main_url = f"{base_url}/%EB%B2%95%EB%A0%B9/%EC%99%B8%EA%B5%AD%EC%9D%B8%EA%B7%BC%EB%A1%9C%EC%9E%90%EC%9D%98%EA%B3%A0%EC%9A%A9%EB%93%B1%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0"

    print("Selenium 설정 시작")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        print(f"메인 URL 접속: {main_url}")
        driver.get(main_url)

        # iframe으로 전환
        wait = WebDriverWait(driver, 10)
        try:
            iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
            driver.switch_to.frame(iframe)
        except TimeoutException:
            print(f"iframe을 찾을 수 없습니다: {main_url}")
            return []

        print("메인 페이지 내용 가져오기 시작")
        main_content = get_content(driver, is_main=True)
        if not main_content:
            print("메인 페이지 내용을 가져오지 못했습니다.")
            return []
        print("메인 페이지 내용 가져오기 완료")
        
        results = [{"url": main_url, "content": main_content}]
        visited_urls = set()  # 중복 링크를 방지하기 위한 set

        print("링크 찾기 시작")
        try:
            links = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[@title='팝업으로 이동']")))
            print(f"찾은 링크 수: {len(links)}")
        except TimeoutException:
            print("링크를 찾을 수 없습니다.")
            return results

        for link in links:
            popup_js = link.get_attribute('onclick')
            print(f"popup_js: {popup_js}")  # 디버깅용으로 추가
            
            if "fncLsLawPop" in popup_js:
                popup_params = popup_js.split("'")
                popup_url = f"https://www.law.go.kr/lsInfoP.do?lsiSeq={popup_params[1]}"
            elif "cptOfiPop" in popup_js:
                popup_params = popup_js.split("'")
                popup_url = popup_params[1]
            else:
                continue  # 인식하지 못하는 onclick 이벤트는 건너뜀

            # URL 중복 여부 확인
            if popup_url in visited_urls:
                print(f"중복된 링크 발견, 건너뜁니다: {popup_url}")
                continue
            
            visited_urls.add(popup_url)  # URL을 방문한 것으로 표시

            print(f"처리 중인 링크: {popup_url}")

            # JavaScript 실행하여 새 창 열기
            try:
                driver.execute_script(popup_js)
                # 새로운 창이 열릴 때까지 기다리기
                WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
                # 새로 열린 창으로 전환
                driver.switch_to.window(driver.window_handles[-1])
                
                # 페이지가 완전히 로드될 때까지 대기
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                content = get_content(driver, is_main=False)
                if content:
                    results.append({"url": popup_url, "content": content})
            except Exception as e:
                print(f"새 창 처리 중 오류 발생: {e}")
            
            # 새 창 닫기
            driver.close()
            
            # 원래 창으로 전환
            driver.switch_to.window(driver.window_handles[0])
            driver.switch_to.frame(iframe)
            time.sleep(1)

        return results

    except Exception as e:
        print(f"스크래핑 중 오류 발생: {e}")
        return []

    finally:
        print("스크래핑 완료")
        driver.quit()

if __name__ == "__main__":
    results = scrape_law_website()
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Content: {result['content']}")  # 내용 일부만 출력
