import json
from selenium.webdriver.common.by import By
from app.models.law_and_visa.law_and_visa_util import wait_for_element, wait_for_element_safely
import time
import re

# 카테고리별로 나누기
def parse_content_category(content):
    sections = {
        "판시사항": re.search(r'【판시사항】(.*?)(?=【|$)', content, re.DOTALL),
        "판결요지": re.search(r'【판결요지】(.*?)(?=【|$)', content, re.DOTALL),
        "참조조문": re.search(r'【참조조문】(.*?)(?=【|$)', content, re.DOTALL),
        "참조판례": re.search(r'【참조판례】(.*?)(?=【|$)', content, re.DOTALL),
        "전문": re.search(r'【전\s*문】(.*?)(?=【|$)', content, re.DOTALL),
        "주문": re.search(r'【주\s*문】(.*?)(?=【|$)', content, re.DOTALL),
        "이유": re.search(r'【이\s*유】(.*?)(?=【|$)', content, re.DOTALL)
    }
    
    parsed_content = {}
    for key, match in sections.items():
        if match:
            # 내용을 추출하고 정리합니다
            section_content = match.group(1).strip()

            # 연속된 줄바꿈을 하나의 공백으로 대체하고, 앞뒤 공백을 제거합니다
            section_content = re.sub(r'\s+', ' ', section_content).strip()
            parsed_content[key] = section_content
    
    return parsed_content

#조항별로 나누기
def parse_law_content(content):
    structure = {}
    current_chapter = None
    current_article = None
    current_content = []
    has_structure = False

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        chapter_match = re.match(r'^제(\d+)장\s+(.+)', line)
        article_match = re.match(r'^제(\d+)조(\(([^)]+)\))?', line)
        
        if chapter_match:
            has_structure = True
            if current_article:
                add_content_to_structure(structure, current_chapter, current_article, current_content)
            current_chapter = f"제{chapter_match.group(1)}장"
            structure[current_chapter] = {"title": chapter_match.group(2), "articles": {}}
            current_article = None
            current_content = []
        elif article_match:
            has_structure = True
            if current_article:
                add_content_to_structure(structure, current_chapter, current_article, current_content)
            current_article = f"제{article_match.group(1)}조"
            if article_match.group(3):
                current_article += f"({article_match.group(3)})"
            current_content = [line]
        elif line.startswith('부칙'):
            has_structure = True
            if current_article:
                add_content_to_structure(structure, current_chapter, current_article, current_content)
            current_chapter = None
            current_article = '부칙'
            current_content = [line]
        else:
            current_content.append(line)

    if has_structure:
        if current_article:
            add_content_to_structure(structure, current_chapter, current_article, current_content)
    else:
        # 구조가 없는 경우 전체 내용을 하나의 content로 처리
        structure = {'content': clean_content(current_content)}

    return structure

# 내용을 정리하여 구조에 추가
def add_content_to_structure(structure, chapter, article, content):
    cleaned_content = clean_content(content)
    if chapter:
        structure[chapter]["articles"][article] = cleaned_content
    else:
        structure[article] = cleaned_content

# 내용 정리: 연속된 줄바꿈을 하나로 줄이고 각 문단을 정리
def clean_content(content):
    paragraphs = [para.strip() for para in ' '.join(content).split('\n') if para.strip()]
    return '\n'.join(paragraphs)

def scrap_law_a(driver, url):
    driver.get(url)
    time.sleep(5)

    print("페이지 로딩 완료")
    
    all_laws = []  # 모든 법률 정보를 저장할 리스트
    content_count = 0 # 내용 수
    while True:
        link_elements = wait_for_element(driver, By.XPATH, "//a[@name='listCont']")
        link_elements = driver.find_elements(By.XPATH, "//a[@name='listCont']")

        for index, link_element in enumerate(link_elements, 1):
            try:
                content_count += 1
                print(f"처리 중인 항목: {index}/{len(link_elements)}")
                
                title = link_element.text.strip()
                print(f"제목: {title}")

                driver.execute_script("arguments[0].click();", link_element)
                time.sleep(3)

                driver.switch_to.window(driver.window_handles[-1])

                print(f"현재 URL: {driver.current_url}")

                content_selectors = [".page_area"]
                content = ""
                
                for selector in content_selectors:
                    try:
                        content_element = wait_for_element(driver, By.CSS_SELECTOR, selector, timeout=5)
                        content = content_element.text
                        content = parse_content_category(content)
                        if content:
                            break
                    except:
                        continue

                if content:
                    print(f"count: {content_count}") 
                    print(f"제목: {title}") 
                    print(f"파싱된 내용: {str(content)}")  # 처음 200자만 출력
                    all_laws.append({"title": title, "content": content})  # 파싱된 법률 정보 저장
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
    with open('parsed_laws_a.json', 'w', encoding='utf-8') as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=4)

    print(f"총 {len(all_laws)}개의 파싱된 법률 정보를 'parsed_laws_a.json' 파일에 저장했습니다.")

    return all_laws

def cscrap_law_b(driver, url):
    driver.get(url)
    time.sleep(10)

    print("페이지 로딩 완료")
    
    all_laws = []
    content_count = 0

    while True:
        link_elements = wait_for_element(driver, By.XPATH, "//a[@name='listCont']")
        if link_elements is None:
            print("링크 요소를 찾을 수 없습니다. 다른 선택자를 시도합니다.")
            link_elements = wait_for_element(driver, By.CSS_SELECTOR, ".search_result a")
        
        if link_elements is None:
            print("링크 요소를 찾을 수 없습니다. 크롤링을 종료합니다.")
            break

        link_elements = driver.find_elements(By.XPATH, "//a[@name='listCont']") or driver.find_elements(By.CSS_SELECTOR, ".search_result a")

        for index, link_element in enumerate(link_elements, 1):
            try:
                content_count += 1
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

                content_element = wait_for_element(driver, By.CLASS_NAME, "page_area")
                if content_element:
                    content = content_element.text
                    parsed_content = parse_law_content(content)
                    print(f"count: {content_count}") 
                    print(f"제목: {title}") 
                    print(f"파싱된 내용: {str(parsed_content)}...")  # 처음 200자만 출력
                    all_laws.append({"title": title, "content": parsed_content})
                else:
                    print("내용을 찾을 수 없습니다.")
                print("-------------------")

                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                
                time.sleep(2)

            except Exception as e:
                print(f"항목 처리 중 오류 발생: {str(e)}")

        next_button = wait_for_element(driver, By.XPATH, "//a/img[@alt='다음 페이지']")
        if next_button:
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)
        else:
            print("더 이상 다음 페이지가 없습니다.")
            break

    # 크롤링한 데이터를 JSON 파일로 저장
    with open('parsed_laws.json', 'w', encoding='utf-8') as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=4)

    print(f"총 {len(all_laws)}개의 파싱된 법률 정보를 'parsed_laws.json' 파일에 저장했습니다.")

    return all_laws