import os
import json
import re
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# 웹 요소를 기다리는 유틸리티 함수
def wait_for_element(driver, by, value, timeout=10):
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))

# 판례 내용을 카테고리별로 파싱하는 함수
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
            section_content = match.group(1).strip()
            section_content = re.sub(r'\s+', ' ', section_content).strip()
            parsed_content[key] = section_content
    
    return parsed_content

# 법령 내용을 구조화하여 파싱하는 함수
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
        structure = {'content': clean_content(current_content)}

    return structure

# 파싱된 내용을 구조에 추가하는 헬퍼 함수
def add_content_to_structure(structure, chapter, article, content):
    cleaned_content = clean_content(content)
    if chapter:
        structure[chapter]["articles"][article] = cleaned_content
    else:
        structure[article] = cleaned_content

# 내용을 정리하는 헬퍼 함수
def clean_content(content):
    paragraphs = [para.strip() for para in ' '.join(content).split('\n') if para.strip()]
    return '\n'.join(paragraphs)

# FAISS 인덱스를 업데이트하는 함수
def update_faiss_index(laws, embeddings, index_name):
    documents = []
    for law in laws:
        content = f"제목: {law['title']}\n\n"
        if isinstance(law['content'], dict):
            for section, section_content in law['content'].items():
                content += f"{section}:\n{section_content}\n\n"
        else:
            content += f"내용:\n{law['content']}\n\n"
        documents.append(Document(page_content=content, metadata={"title": law['title']}))
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,  # chunk_size를 줄임
        chunk_overlap=100,  # chunk_overlap을 줄임
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f"faiss_index_{index_name}")
    
    if os.path.exists(save_path):
        print(f"기존 FAISS 인덱스를 업데이트합니다: {save_path}")
        vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(texts)
    else:
        print(f"새로운 FAISS 인덱스를 생성합니다: {save_path}")
        vector_store = FAISS.from_documents(texts, embeddings)
    
    vector_store.save_local(save_path)
    print(f"FAISS 인덱스를 저장했습니다: {save_path}")
    
    return len(texts)


# 판례 정보를 스크래핑하는 함수
def scrap_law_a(driver, url, embeddings, index_name):
    driver.get(url)
    time.sleep(5)
    print("페이지 로딩 완료")
    
    all_laws = []
    content_count = 0

    while True:
        page_laws = []
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
                    print(f"파싱된 내용: {str(content)[:200]}...")  # 처음 200자만 출력
                    page_laws.append({"title": title, "content": content})
                else:
                    print("내용을 찾을 수 없습니다.")
                print("-------------------")

                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(1)

            except Exception as e:
                print(f"항목 처리 중 오류 발생: {str(e)}")

        # 현재 페이지의 법률 정보를 FAISS에 저장
        update_faiss_index(page_laws, embeddings, f"{index_name}_page")
        all_laws.extend(page_laws)

        try:
            next_button = wait_for_element(driver, By.XPATH, "//a/img[@alt='다음 페이지']", timeout=10)
            if next_button:
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(1)
            else:
                print("더 이상 다음 페이지가 없습니다.")
                break
        except TimeoutException:
            print("다음 페이지 버튼을 찾을 수 없습니다. 크롤링을 종료합니다.")
            break

    # # 크롤링한 데이터를 JSON 파일로 저장
    # with open('parsed_laws_a.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_laws, f, ensure_ascii=False, indent=4)

    # print(f"총 {len(all_laws)}개의 파싱된 법률 정보를 'parsed_laws_a.json' 파일에 저장했습니다.")

    return all_laws

# 법령, 조약, 규칙/예규/선례 정보를 스크래핑하는 함수
def scrap_law_b(driver, url, embeddings, index_name):
    driver.get(url)
    time.sleep(10)

    print("페이지 로딩 완료")
    
    all_laws = []
    content_count = 0

    while True:
        page_laws = []
        link_elements = wait_for_element(driver, By.XPATH, "//a[@name='listCont']")
        if link_elements is None:
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
                time.sleep(1)

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
                    print(f"파싱된 내용: {str(parsed_content)[:200]}...")  # 처음 200자만 출력
                    page_laws.append({"title": title, "content": parsed_content})
                else:
                    print("내용을 찾을 수 없습니다.")
                print("-------------------")

                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                
                time.sleep(2)

            except Exception as e:
                print(f"항목 처리 중 오류 발생: {str(e)}")

        # 현재 페이지의 법률 정보를 FAISS에 저장
        update_faiss_index(page_laws, embeddings, f"{index_name}_page")
        all_laws.extend(page_laws)

        next_button = wait_for_element(driver, By.XPATH, "//a/img[@alt='다음 페이지']")
        if next_button:
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(1)
        else:
            print("더 이상 다음 페이지가 없습니다.")
            break

    # # 크롤링한 데이터를 JSON 파일로 저장
    # with open(f'parsed_laws_{index_name}.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_laws, f, ensure_ascii=False, indent=4)

    # print(f"총 {len(all_laws)}개의 파싱된 법률 정보를 'parsed_laws_{index_name}.json' 파일에 저장했습니다.")

    return all_laws