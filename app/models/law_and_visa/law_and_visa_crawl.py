import json
from selenium.webdriver.common.by import By
from app.models.law_and_visa.law_and_visa_util import wait_for_element, wait_for_element_safely
import time

def crawl_law_a(driver, url):
    driver.get(url)
    time.sleep(5)

    print("페이지 로딩 완료")
    
    all_laws = []  # 모든 법률 정보를 저장할 리스트

    while True:
        link_elements = wait_for_element(driver, By.XPATH, "//a[@name='listCont']")
        link_elements = driver.find_elements(By.XPATH, "//a[@name='listCont']")

        for index, link_element in enumerate(link_elements, 1):
            try:
                print(f"처리 중인 항목: {index}/{len(link_elements)}")
                
                title = link_element.text.strip()
                print(f"제목: {title}")

                driver.execute_script("arguments[0].click();", link_element)
                time.sleep(3)

                driver.switch_to.window(driver.window_handles[-1])

                print(f"현재 URL: {driver.current_url}")

                content_selectors = [".page_area", "#totalTxt", "#content", ".content", "body"]
                content = ""
                for selector in content_selectors:
                    try:
                        content_element = wait_for_element(driver, By.CSS_SELECTOR, selector, timeout=5)
                        content = content_element.text
                        if content:
                            break
                    except:
                        continue

                if content:
                    print(f"내용: {content[:200]}...")  # 처음 200자만 출력
                    all_laws.append({"title": title, "content": content})  # 법률 정보 저장
                else:
                    print("내용을 찾을 수 없습니다.")
                print("-------------------")

                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(2)

            except Exception as e:
                print(f"항목 처리 중 오류 발생: {str(e)}")

        try:
            next_button = wait_for_element(driver, By.XPATH, "//a/img[@alt='다음 페이지']")
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(3)
        except:
            print("더 이상 다음 페이지가 없습니다.")
            break

    # 크롤링한 데이터를 JSON 파일로 저장
    with open('crawled_laws_a.json', 'w', encoding='utf-8') as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=4)

    print(f"총 {len(all_laws)}개의 법률 정보를 크롤링하여 'crawled_laws_a.json' 파일에 저장했습니다.")

    return all_laws

def crawl_law_b(driver, url):
    driver.get(url)
    time.sleep(10)

    print("페이지 로딩 완료")
    
    all_laws = []  # 모든 법률 정보를 저장할 리스트

    while True:
        link_elements = wait_for_element_safely(driver, By.XPATH, "//a[@name='listCont']")
        if link_elements is None:
            print("링크 요소를 찾을 수 없습니다. 다른 선택자를 시도합니다.")
            link_elements = wait_for_element_safely(driver, By.CSS_SELECTOR, ".search_result a")
        
        if link_elements is None:
            print("링크 요소를 찾을 수 없습니다. 크롤링을 종료합니다.")
            break

        link_elements = driver.find_elements(By.XPATH, "//a[@name='listCont']") or driver.find_elements(By.CSS_SELECTOR, ".search_result a")

        for index, link_element in enumerate(link_elements, 1):
            try:
                print(f"처리 중인 항목: {index}/{len(link_elements)}")
                
                title = link_element.text.strip()
                print(f"제목: {title}")

                driver.execute_script("arguments[0].click();", link_element)
                time.sleep(5)

                if len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                else:
                    print("새 창이 열리지 않았습니다.")

                print(f"현재 URL: {driver.current_url}")

                content_selectors = [".page_area", "#totalTxt", "#content", ".content", "body"]
                content = ""
                for selector in content_selectors:
                    content_element = wait_for_element_safely(driver, By.CSS_SELECTOR, selector, timeout=5)
                    if content_element:
                        content = content_element.text
                        break

                if content:
                    print(f"내용: {content[:200]}...")  # 처음 200자만 출력
                    all_laws.append({"title": title, "content": content})  # 법률 정보 저장
                else:
                    print("내용을 찾을 수 없습니다.")
                print("-------------------")

                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                
                time.sleep(2)

            except Exception as e:
                print(f"항목 처리 중 오류 발생: {str(e)}")

        next_button = wait_for_element_safely(driver, By.XPATH, "//a/img[@alt='다음 페이지']")
        if next_button:
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)
        else:
            print("더 이상 다음 페이지가 없습니다.")
            break

    # 크롤링한 데이터를 JSON 파일로 저장
    with open('crawled_laws_b.json', 'w', encoding='utf-8') as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=4)

    print(f"총 {len(all_laws)}개의 법률 정보를 크롤링하여 'crawled_laws_b.json' 파일에 저장했습니다.")

    return all_laws