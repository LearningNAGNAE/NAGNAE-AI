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
from langchain.text_splitter import RecursiveCharacterTextSplitter



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

                # contId 추출
                cont_id = None
                current_url = driver.current_url
                cont_id_match = re.search(r'contId=(\d+)', current_url)
                if cont_id_match:
                    cont_id = cont_id_match.group(1)
                else:
                    print(f"contId를 찾을 수 없습니다. URL: {current_url}")

                print(f"현재 URL: {current_url}")
                print(f"추출된 contId: {cont_id}")

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
                    print(f"내용: {str(content)[:200]}...")  # 처음 200자만 출력
                    print(f"contId: {cont_id}")
                    
                    section_count = 0
                    for section, section_content in content.items():
                        section_count += 1

                        # 각 섹션에 대해 별도의 임베딩 생성
                        text_to_embed = f"제목: {title}\n섹션: {section}\n\n내용: {section_content}"
                        embedding = create_embedding(text_to_embed, embeddings)
                        
                        page_laws.append({
                            "cont_id": cont_id,
                            "title": title,
                            "section": section,
                            "content": section_content,
                            "embedding": embedding
                        })
                    print(f"총 {section_count}개의 섹션이 처리되었습니다.")
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

    return all_laws


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

                current_url = driver.current_url
                print(f"현재 URL: {current_url}")

                # contId 추출
                cont_id = None
                cont_id_match = re.search(r'contId=(\d+)', current_url)
                if cont_id_match:
                    cont_id = cont_id_match.group(1)
                else:
                    print(f"contId를 찾을 수 없습니다. URL: {current_url}")

                content_element = wait_for_element(driver, By.CLASS_NAME, "page_area")
                if content_element:
                    content = content_element.text
                    parsed_content = parse_law_content(content)
                    print(f"count: {content_count}") 
                    print(f"제목: {title}") 
                    print(f"파싱된 내용: {str(parsed_content)[:200]}...")  # 처음 200자만 출력
                    
                    for article, article_content in parsed_content.items():
                        # 각 조항에 대해 별도의 임베딩 생성
                        text_to_embed = f"제목: {title}\n조항: {article}\n\n내용: {article_content}"
                        embedding = create_embedding(text_to_embed, embeddings)
                        
                        page_laws.append({
                            "cont_id": cont_id,
                            "title": title,
                            "article": article,
                            "content": article_content,
                            "embedding": embedding
                        })
                    
                    print(f"총 {len(parsed_content)}개의 조항이 처리되었습니다.")
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

    return all_laws


def scrap_law_j(driver, url, embeddings, index_name):
    print(f"URL 접속: {url}")
    driver.get(url)
    time.sleep(1)
    print("페이지 로딩 완료")

    all_laws = []
    processed_links = set()  # 처리된 링크를 추적하기 위한 집합

    try:
        # iframe으로 전환
        wait = WebDriverWait(driver, 10)
        try:
            iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
            driver.switch_to.frame(iframe)
        except TimeoutException:
            print(f"iframe을 찾을 수 없습니다: {url}")
            return []

        print("메인 페이지 내용 가져오기 시작")
        main_content = get_content(driver)
        if not main_content:
            print("메인 페이지 내용을 가져오지 못했습니다.")
            return []
        print("메인 페이지 내용 가져오기 완료")
        
        # 메인 페이지 내용을 파싱하여 조항별로 나눕니다.
        parsed_content = parse_law_content_j(main_content)
        
        for article, article_content in parsed_content.items():
            # 각 조항에 대해 별도의 임베딩 생성
            text_to_embed = f"제목: 외국인노동법\n조항: {article}\n\n내용: {article_content}"
            embedding = create_embedding(text_to_embed, embeddings)
            
            all_laws.append({
                "title": "외국인노동법",
                "article": article,
                "content": article_content,
                "embedding": embedding
            })

        print(f"메인 페이지에서 총 {len(all_laws)}개의 조항이 처리되었습니다.")

        # 링크 찾기 및 처리
        print("링크 찾기 시작")
        try:
            links = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[@title='팝업으로 이동']")))
            print(f"찾은 링크 수: {len(links)}")
        except TimeoutException:
            print("링크를 찾을 수 없습니다.")
            return all_laws

        for index, link in enumerate(links, 1):
            popup_js = link.get_attribute('onclick')
            print(f"처리 중인 항목: {index}/{len(links)}")
            print(f"popup_js: {popup_js}")
            
            if "fncLsLawPop" in popup_js:
                popup_params = popup_js.split("'")
                popup_url = f"https://www.law.go.kr/lsInfoP.do?lsiSeq={popup_params[1]}"
            elif "cptOfiPop" in popup_js:
                popup_params = popup_js.split("'")
                popup_url = popup_params[1]
            else:
                continue

            print(f"처리 중인 링크: {popup_url}")

            try:
                driver.execute_script(popup_js)
                WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
                driver.switch_to.window(driver.window_handles[-1])
                
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                popup_content = get_content(driver)
                print(f"팝업 내용 길이: {len(popup_content)}")
                print(f"팝업 내용 미리보기: {popup_content[:200]}...")  # 처음 200자만 출력
                
                if popup_content:
                    popup_parsed_content = parse_law_content_j(popup_content)
                    print(f"파싱된 조항 수: {len(popup_parsed_content)}")
                    for popup_article, popup_article_content in popup_parsed_content.items():
                        print(f"조항: {popup_article}")
                        print(f"내용 미리보기: {popup_article_content[:100]}...")  # 처음 100자만 출력
                        
                        text_to_embed = f"제목: 외국인노동법 추가정보\n조항: {popup_article}\n\n내용: {popup_article_content}"
                        embedding = create_embedding(text_to_embed, embeddings)
                        
                        all_laws.append({
                            "title": "외국인노동법 추가정보",
                            "article": popup_article,
                            "content": popup_article_content,
                            "embedding": embedding
                        })
                else:
                    print("팝업 내용이 비어 있습니다.")
                
                print(f"팝업 {index} 처리 완료")
            except Exception as e:
                print(f"새 창 처리 중 오류 발생: {e}")
            
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            driver.switch_to.frame(iframe)
            time.sleep(1)

        print(f"총 {len(all_laws)}개의 항목이 처리되었습니다.")
        print(f"처리된 고유 링크 수: {len(processed_links)}")

        # 현재 페이지의 법률 정보를 FAISS에 저장
        update_faiss_index(all_laws, embeddings, f"{index_name}_page")

        return all_laws
    except Exception as e:
        print(f"scrap_law_j 함수 실행 중 예외 발생: {e}")
        return all_laws

# --------------------------------------------------------------------------------- #
# 기존 코드에서 가져온 get_content 함수
def get_content(driver):
    content = ""
    try:
        content_element = driver.find_element(By.TAG_NAME, "body")
        content = content_element.text
    except Exception as e:
        print(f"콘텐츠 가져오기 중 오류 발생: {e}")
    return content


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

def parse_law_content_j(content):
    articles = {}
    current_article = None
    current_content = []

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        article_match = re.match(r'^제(\d+)조(\(([^)]+)\))?', line)
        
        if article_match:
            if current_article:
                articles[current_article] = '\n'.join(current_content)
            current_article = f"제{article_match.group(1)}조"
            if article_match.group(3):
                current_article += f"({article_match.group(3)})"
            current_content = [line]
        else:
            current_content.append(line)

    if current_article:
        articles[current_article] = '\n'.join(current_content)

    return articles

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    for law in laws:
        title = law['title']
        cont_id = law.get('cont_id')
        section = law.get('section')
        article = law.get('article')
        content = law['content']

        if section:  # scrap_law_a의 경우
            full_content = f"제목: {title}\n섹션: {section}\n\n{content}"
        elif article:  # scrap_law_b의 경우
            full_content = f"제목: {title}\n조항: {article}\n\n{content}"
        else:  # 기타 경우
            full_content = f"제목: {title}\n\n{content}"

        # 긴 내용을 청크로 나눕니다
        chunks = text_splitter.split_text(full_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "title": title,
                    "section": section,
                    "article": article,
                    "cont_id": cont_id,
                    "chunk_index": i
                }
            )
            documents.append(doc)

    print(f"총 {len(documents)}개의 청크가 생성되었습니다.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f"faiss_index_{index_name}")
    
    if os.path.exists(save_path):
        print(f"기존 FAISS 인덱스 업데이트: {save_path}")
        vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        print(f"새로운 FAISS 인덱스 생성: {save_path}")
        vector_store = FAISS.from_documents(documents, embeddings)
    
    vector_store.save_local(save_path)
    print(f"FAISS 인덱스 저장: {save_path}")
    
    return len(documents)

# 임베딩 함수
def create_embedding(text, embeddings):
    embedding = embeddings.embed_query(text)
    return embedding


#청크 분할 개선
def create_text_splitter():
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

