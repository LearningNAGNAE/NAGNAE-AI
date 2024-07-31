import openai
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

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
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 고유 ID 필드 추가
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Law Embeddings Collection")
    
<<<<<<< HEAD
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



import time
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from typing import List, Dict, Any

def setup_driver():
    # Selenium WebDriver 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    return webdriver.Chrome(options=options)

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

def update_faiss_index(laws: List[Dict[str, Any]], embeddings: Any, index_name: str) -> int:
    if not laws:
        print(f"업데이트할 법률 정보가 없습니다: {index_name}")
        return 0

    vectors = [law['embedding'] for law in laws if 'embedding' in law]
    
    if not vectors:
        print(f"유효한 임베딩이 없습니다: {index_name}")
        return 0

    vectors_np = np.array(vectors).astype('float32')
    dimension = vectors_np.shape[1]

    try:
        index = faiss.read_index(f"{index_name}.index")
        print(f"기존 인덱스를 로드했습니다: {index_name}")
    except:
        index = faiss.IndexFlatL2(dimension)
        print(f"새 인덱스를 생성했습니다: {index_name}")

    index.add(vectors_np)
    faiss.write_index(index, f"{index_name}.index")
    
    print(f"인덱스가 업데이트되었습니다: {index_name}, 총 {index.ntotal}개의 벡터")
    
    return index.ntotal

# -- b --
def scrap_law_b(driver, url, embeddings, index_name):
    driver.get(url)
    time.sleep(5)  # 초기 페이지 로딩 시간

    print("초기 페이지 로딩 완료")
    
    # 53페이지로 이동
    page_reached = False
    attempts = 0
    max_attempts = 3

    while not page_reached and attempts < max_attempts:
        attempts += 1
        print(f"53페이지로 이동 시도 {attempts}/{max_attempts}")
        
        try:
            # 페이지 입력 필드 찾기
            page_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input.go_page"))
            )
            page_input.clear()
            page_input.send_keys("53")

            # '페이지 이동' 버튼 클릭
            move_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn_move"))
            )
            driver.execute_script("arguments[0].click();", move_button)
            time.sleep(5)

            # 페이지 이동 확인
            page_info = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "p.list_location"))
            )
            current_page = page_info.text.split('/')[0].strip()
            if current_page == "53":
                page_reached = True
                print("53페이지 도달 성공")
            else:
                print(f"현재 페이지: {current_page}. 53페이지 도달 실패")
        
        except Exception as e:
            print(f"53페이지로 이동 중 오류 발생: {str(e)}")

    if not page_reached:
        print("53페이지로 이동 실패. 현재 페이지에서 크롤링을 시작합니다.")

    all_laws = []
    content_count = 0
    current_page = 53 if page_reached else int(current_page)

    while True:
        print(f"현재 처리 중인 페이지: {current_page}")
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
                time.sleep(3)  # 클릭 후 대기 시간

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

        # 다음 페이지로 이동
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a/img[@alt='다음 페이지']"))
            )
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)  # 페이지 전환 대기 시간 증가
            current_page += 1
        except TimeoutException:
            print("더 이상 다음 페이지가 없습니다.")
            break
        except Exception as e:
            print(f"다음 페이지로 이동 중 오류 발생: {str(e)}")
            break

    print(f"총 {len(all_laws)}개의 법률 정보가 처리되었습니다.")
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
=======
    # 컬렉션 생성
    collection = Collection(name=collection_name, schema=schema)
    return collection
>>>>>>> stage

# 임베딩
def create_embeddings(laws):
    embeddings = []
    for law in laws:
        embedding = get_embedding(law['content'])
        embeddings.append({'embedding': embedding})
    return embeddings


# 인덱스 생성
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

# 크롤링된 데이터 로드 및 임베딩 생성
def main():
    setup_milvus()
    
    collection_name = "law_embeddings"
    dim = 1536  # text-embedding-ada-002 모델의 임베딩 차원
    
    # 컬렉션이 존재하지 않는 경우 생성
    drop_collection_if_exists(collection_name)
    create_collection(collection_name, dim)
    create_index(collection_name)
    
    # JSON 파일에서 법률 정보 로드
    with open('crawled_laws_a.json', 'r', encoding='utf-8') as f:
        laws_a = json.load(f)
    with open('crawled_laws_b.json', 'r', encoding='utf-8') as f:
        laws_b = json.load(f)
    
    all_laws = laws_a + laws_b
    
    # 임베딩 생성 및 저장
    embeddings = [get_embedding(law['content']) for law in all_laws]
    validate_embeddings([{"embedding": emb} for emb in embeddings], dim)
    
    collection = Collection(name=collection_name)
    save_embeddings(collection, embeddings, all_laws)
    
    # 컬렉션 로드
    load_collection(collection_name)
    
    # 검색 기능 테스트
    query_text = "외국인"
    query_embedding = get_embedding(query_text)
    results = search_similar_documents(collection, query_embedding)
    
    print(f'\n"{query_text}"와 유사한 상위 5개 문서:')
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 제목: {result.entity.get('title')}")
        print(f"   유사도 점수: {result.score:.4f}")
        print(f"   내용 일부: {result.entity.get('content')[:200]}...")

if __name__ == "__main__":
    main()
