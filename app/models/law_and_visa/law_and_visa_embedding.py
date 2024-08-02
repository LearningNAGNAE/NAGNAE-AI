import os
import json
import re
import time
from dotenv import load_dotenv
import openai
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from typing import List, Dict, Any
from app.models.law_and_visa.law_and_visa_util import setup_driver, wait_for_element, wait_for_element_safely
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Milvus 연결 설정
def setup_milvus():
    connections.connect("default", host="localhost", port="19530")

# 컬렉션 생성
def create_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Law Embeddings Collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"컬렉션 '{collection_name}'이(가) 생성되었습니다.")
    return collection


# 판례 정보를 스크래핑하는 함수 a
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


# -- b --
def scrap_law_b(driver, url, embeddings, index_name):
    driver.get(url)
    time.sleep(1)  # 페이지 로딩 시간 증가

    print("페이지 로딩 완료")
    
    all_laws = []
    content_count = 0

    while True:
        page_laws = []
        try:
            link_elements = WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[@name='listCont']"))
            )
        except TimeoutException:
            print("링크 요소를 찾을 수 없습니다. 다음 방법을 시도합니다.")
            try:
                link_elements = WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".search_result a"))
                )
            except TimeoutException:
                print("링크 요소를 찾을 수 없습니다. 크롤링을 종료합니다.")
                break

        for index, link_element in enumerate(link_elements, 1):
            try:
                content_count += 1
                print(f"처리 중인 항목: {index}/{len(link_elements)}")
                
                title = link_element.text.strip()
                print(f"제목: {title}")

                driver.execute_script("arguments[0].click();", link_element)
                time.sleep(3)  # 클릭 후 대기 시간 증가

                # 새 창으로 전환
                driver.switch_to.window(driver.window_handles[-1])
                
                current_url = driver.current_url
                print(f"현재 URL: {current_url}")

                # contId 추출
                cont_id = None
                cont_id_match = re.search(r'contId=(\d+)', current_url)
                if cont_id_match:
                    cont_id = cont_id_match.group(1)
                else:
                    print(f"contId를 찾을 수 없습니다. URL: {current_url}")

                try:
                    content_element = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "page_area"))
                    )
                    content = content_element.text
                    parsed_content = parse_law_content(content)
                    print(f"count: {content_count}") 
                    print(f"제목: {title}") 
                    print(f"파싱된 내용: {str(parsed_content)[:200]}...")  # 처음 200자만 출력
                    
                    for article, article_content in parsed_content.items():
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
                except TimeoutException:
                    print("내용을 찾을 수 없습니다.")
                print("-------------------")

                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(2)

            except StaleElementReferenceException:
                print(f"항목 {index}에 대한 참조가 오래되었습니다. 다음 항목으로 넘어갑니다.")
                continue
            except Exception as e:
                print(f"항목 처리 중 오류 발생: {str(e)}")

        # 현재 페이지의 법률 정보를 FAISS에 저장
        update_faiss_index(page_laws, embeddings, f"{index_name}_page")
        all_laws.extend(page_laws)

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a/img[@alt='다음 페이지']"))
            )
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(3)
        except TimeoutException:
            print("더 이상 다음 페이지가 없습니다.")
            break

    return all_laws


# --- j ---
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
    

# -- 크롤링 --
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def crawl_law_a(url: str, driver: Any, embeddings: OpenAIEmbeddings, index_name: str) -> tuple[List[Dict[str, str]], int]:
    logger.info(f"URL 접속: {url}")
    driver.get(url)
    time.sleep(5)
    logger.info("페이지 로딩 완료")

    crawled_data = []
    processed_links = set()

    try:
        # 메인 페이지 내용 크롤링
        main_title = extract_title(driver)
        main_content = extract_content(driver)
        if main_content:
            crawled_data.append({
                'url': url,
                'title': main_title,
                'content': main_content
            })
            logger.info(f"메인 페이지 제목: {main_title}")
            logger.info(f"메인 페이지 내용: {main_content[:200]}...")

        # 모든 관련 링크 찾기
        links = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/portal/foreigner/ko/"]')
        unique_links = set(link.get_attribute('href') for link in links if link.get_attribute('href'))

        for link in unique_links:
            if link in processed_links:
                continue

            logger.info(f"링크 접속: {link}")
            driver.get(link)
            time.sleep(3)
            processed_links.add(link)

            title = extract_title(driver)
            content = extract_content(driver)

            if content:
                crawled_data.append({
                    'url': link,
                    'title': title,
                    'content': content
                })
                logger.info(f"제목: {title}")
                logger.info(f"내용: {content[:200]}...")  # 처음 200자만 출력
            else:
                logger.warning(f"내용을 찾을 수 없습니다: {link}")

        logger.info(f"총 {len(crawled_data)}개의 항목이 처리되었습니다.")

        if not crawled_data:
            logger.warning("크롤링된 데이터가 없습니다. 웹사이트 구조를 확인해주세요.")
            return [], 0

        # FAISS 인덱스 업데이트
        logger.info("FAISS 인덱스 업데이트를 시작합니다.")
        total_vectors = update_faiss_index(crawled_data, embeddings, f"{index_name}_page")
        logger.info(f"FAISS 인덱스 업데이트가 완료되었습니다. 총 {total_vectors}개의 벡터가 저장되었습니다.")

        return crawled_data, total_vectors

    except Exception as e:
        logger.error(f"crawl_law_a 함수 실행 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], 0

def extract_title(driver):
    title_selectors = ['h2', 'h1', 'h3', '.title', '.tit', '.sub-title']
    for selector in title_selectors:
        try:
            title_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for title_element in title_elements:
                title = title_element.text.strip()
                if title:
                    logger.info(f"제목 추출 성공 (선택자: {selector}): {title}")
                    return title
            logger.warning(f"제목 요소를 찾았으나 내용이 비어 있습니다 (선택자: {selector})")
        except Exception as e:
            logger.warning(f"제목 추출 실패 (선택자: {selector}): {str(e)}")
            continue
    
    # 모든 텍스트 내용을 가져와서 첫 번째 의미 있는 라인을 제목으로 사용
    try:
        body_text = driver.find_element(By.TAG_NAME, 'body').text
        lines = body_text.split('\n')
        for line in lines:
            if line.strip():
                logger.info(f"본문에서 추출한 제목: {line.strip()}")
                return line.strip()
    except Exception as e:
        logger.warning(f"본문에서 제목 추출 실패: {str(e)}")
    
    logger.warning("모든 방법으로 제목을 찾지 못했습니다. 페이지 URL을 사용합니다.")
    return driver.current_url

from selenium.common.exceptions import TimeoutException, NoSuchElementException
def extract_content(driver):
    content_selectors = [
        '.cont-group', '.content', '.txt_cont', 
        'article', 'section', '.main-content'
    ]
    for selector in content_selectors:
        try:
            element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            
            formatted_content = []
            
            # 모든 <li> 태그 찾기
            li_elements = element.find_elements(By.TAG_NAME, 'li')
            for li in li_elements:
                try:
                    dt = li.find_element(By.TAG_NAME, 'dt')
                    dd = li.find_element(By.TAG_NAME, 'dd')
                    dt_text = dt.text.strip()
                    dd_text = dd.text.strip()
                    formatted_content.append(f"{dt_text}: {dd_text}")
                except NoSuchElementException:
                    # <dt>와 <dd>가 없는 경우 그냥 텍스트 추가
                    li_text = li.text.strip()
                    if li_text:
                        formatted_content.append(li_text)
            
            # <li> 태그 외의 다른 텍스트도 포함
            other_elements = element.find_elements(By.XPATH, './/*[not(self::li) and not(self::dt) and not(self::dd)]')
            for elem in other_elements:
                if elem.text.strip():
                    formatted_content.append(elem.text.strip())
            
            return '\n'.join(formatted_content)

        except TimeoutException:
            continue
    return ""

def update_faiss_index(documents: List[Dict[str, str]], embeddings: OpenAIEmbeddings, index_name: str) -> int:
    if not documents:
        logger.warning("문서가 없어 FAISS 인덱스를 업데이트하지 않습니다.")
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    all_chunks = []
    for doc in documents:
        text = f"{doc['title']}\n\n{doc['content']}"
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    # 임베딩 생성
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in all_chunks]

    # FAISS 인덱싱
    embedding_matrix = np.array(chunk_embeddings).astype('float32')
    dimension = embedding_matrix.shape[1]

    try:
        index = faiss.read_index(f"{index_name}.index")
        logger.info(f"기존 인덱스를 로드했습니다: {index_name}")
    except:
        index = faiss.IndexFlatL2(dimension)
        logger.info(f"새 인덱스를 생성했습니다: {index_name}")

    index.add(embedding_matrix)
    faiss.write_index(index, f"{index_name}.index")
    
    logger.info(f"인덱스가 업데이트되었습니다: {index_name}, 총 {index.ntotal}개의 벡터")
    
    return index.ntotal

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    all_chunks = []
    for law in laws:
        text = f"{law['title']}\n\n{law['content']}"
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    # 임베딩 생성
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in all_chunks]

    # FAISS 인덱싱
    embedding_matrix = np.array(chunk_embeddings).astype('float32')
    dimension = embedding_matrix.shape[1]

    try:
        index = faiss.read_index(f"{index_name}.index")
        print(f"기존 인덱스를 로드했습니다: {index_name}")
    except:
        index = faiss.IndexFlatL2(dimension)
        print(f"새 인덱스를 생성했습니다: {index_name}")

    index.add(embedding_matrix)
    faiss.write_index(index, f"{index_name}.index")
    
    print(f"인덱스가 업데이트되었습니다: {index_name}, 총 {index.ntotal}개의 벡터")
    
    return index.ntotal

# 임베딩 생성 함수
def create_embeddings(laws):
    embeddings = []
    for law in laws:
        embedding = get_embedding(law['content'])
        embeddings.append({'embedding': embedding})
    return embeddings


# 인덱스 생성 함수
def create_index(collection_name):
    collection = Collection(name=collection_name)
    # 인덱스 생성 (예: IVF_FLAT)
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
        "metric_type": "L2"
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"인덱스가 컬렉션 '{collection_name}'에 생성되었습니다.")

# OpenAI API를 사용하여 임베딩 생성
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 임베딩 데이터 검증
def validate_embeddings(embeddings, dim):
    for emb in embeddings:
        if not isinstance(emb["embedding"], list) or len(emb["embedding"]) != dim:
            raise ValueError(f"임베딩 데이터가 리스트 형식이 아니거나 {dim} 차원이 아닙니다.")
        if not all(isinstance(x, float) for x in emb["embedding"]):
            raise ValueError("임베딩 데이터에 부동소수점(float) 타입이 아닌 값이 포함되어 있습니다.")

# 임베딩 저장
def save_embeddings(collection, embeddings, laws):
    try:
        # IDs와 임베딩 벡터 추출
        ids = [i for i in range(len(embeddings))]  # 고유 ID 생성
        embedding_vectors = [emb['embedding'] for emb in embeddings]

        # 데이터 삽입
        collection.insert([
            {"name": "id", "type": DataType.INT64, "values": ids},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "values": embedding_vectors}
        ])
        print("임베딩 벡터가 컬렉션에 삽입되었습니다.")
    except Exception as e:
        print(f"예외 발생: {e}")
        raise


# 컬렉션 삭제
def drop_collection_if_exists(collection_name):
    if utility.has_collection(collection_name):
        print(f"컬렉션 '{collection_name}'이(가) 이미 존재하므로 삭제합니다.")
        utility.drop_collection(collection_name)
        
# 컬렉션 로드
def load_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.load()
    print(f"컬렉션 '{collection_name}'이(가) 메모리에 로드되었습니다.")

# 컬렉션 스키마 확인
def check_collection_schema(collection_name):
    collection = Collection(name=collection_name)
    schema = collection.schema
    print("컬렉션 스키마:")
    for field in schema.fields:
        print(f"Field name: {field.name}, Type: {field.dtype}, Dim: {field.dim}")

# 유사한 문서 검색
def search_similar_documents(collection, query_embedding):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5
    )
    return results[0]

def create_embedding(text: str, embeddings: OpenAIEmbeddings) -> List[float]:
    return embeddings.embed_query(text)

def parse_law_content(content: str) -> Dict[str, str]:
    # 법률 내용을 파싱하는 로직 (예시)
    # 실제 구현은 웹사이트의 구조에 따라 달라질 수 있습니다
    articles = {}
    current_article = ""
    for line in content.split('\n'):
        if line.startswith('제') and '조' in line:
            current_article = line
            articles[current_article] = ""
        elif current_article:
            articles[current_article] += line + "\n"
    return articles
