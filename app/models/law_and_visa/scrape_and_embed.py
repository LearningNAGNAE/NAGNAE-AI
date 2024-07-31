import os
import time
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from httpcore import TimeoutException
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Selenium을 사용한 법률 데이터 스크래핑
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
        visited_urls = set([main_url])

        print("링크 찾기 시작")
        try:
            links = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[@title='팝업으로 이동']")))
            print(f"찾은 링크 수: {len(links)}")
        except TimeoutException:
            print("링크를 찾을 수 없습니다.")
            return results

        for link in links:
            popup_js = link.get_attribute('onclick')
            print(f"popup_js: {popup_js}")
            
            if "fncLsLawPop" in popup_js:
                popup_params = popup_js.split("'")
                popup_url = f"https://www.law.go.kr/lsInfoP.do?lsiSeq={popup_params[1]}"
            elif "cptOfiPop" in popup_js:
                popup_params = popup_js.split("'")
                popup_url = popup_params[1]
            else:
                continue

            if popup_url in visited_urls:
                print(f"중복된 링크 발견, 건너뜁니다: {popup_url}")
                continue
            
            visited_urls.add(popup_url)

            print(f"처리 중인 링크: {popup_url}")

            try:
                driver.execute_script(popup_js)
                WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
                driver.switch_to.window(driver.window_handles[-1])
                
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                content = get_content(driver, is_main=False)
                if content:
                    results.append({"url": popup_url, "content": content})
            except Exception as e:
                print(f"새 창 처리 중 오류 발생: {e}")
            
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            driver.switch_to.frame(iframe)
            time.sleep(1)

        return results
    finally:
        driver.quit()

# Selenium을 사용하여 콘텐츠 가져오기
def get_content(driver, is_main=False):
    content = ""
    try:
        content_element = driver.find_element(By.TAG_NAME, "body")
        content = content_element.text
    except Exception as e:
        print(f"콘텐츠 가져오기 중 오류 발생: {e}")
    return content

# 메인 함수
def main():
    try:
        print("법률 데이터 스크래핑 시작")
        scraped_data = scrape_law_website()
        if not scraped_data:
            print("스크래핑된 데이터가 없습니다.")
            return

        # 텍스트 분할
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = [
            Document(page_content=doc["content"], metadata={"source": doc["url"]})
            for doc in scraped_data
        ]
        documents = text_splitter.split_documents(texts)

        # 임베딩 생성 및 Faiss 인덱스 생성
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        # 대화형 검색 체인 생성
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        # 대화 루프
        while True:
            query = input("질문을 입력하세요 (종료하려면 'q' 입력): ")
            if query.lower() == 'q':
                break

            result = qa({"question": query})
            print(f"답변: {result['answer']}")

    except Exception as e:
        print(f"메인 함수 실행 중 예외 발생: {e}")

# 메인 함수 실행
if __name__ == "__main__":
    main()