import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import concurrent.futures
import schedule
import time
from datetime import datetime, timedelta
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models.law_and_visa.law_and_visa_util import setup_driver
from app.models.law_and_visa.law_and_visa_embedding import scrap_law_a, scrap_law_b, scrap_law_j, crawl_law_a

load_dotenv()

LAST_UPDATE_FILE = "last_update.txt"
SCRAPED_CONT_IDS_FILE = "scraped_cont_ids.json"

def get_last_update_time():
    if os.path.exists(LAST_UPDATE_FILE):
        with open(LAST_UPDATE_FILE, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    return datetime.min

def set_last_update_time(dt):
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(dt.isoformat())

def load_scraped_cont_ids():
    if os.path.exists(SCRAPED_CONT_IDS_FILE):
        with open(SCRAPED_CONT_IDS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_scraped_cont_ids(cont_ids):
    with open(SCRAPED_CONT_IDS_FILE, "w") as f:
        json.dump(cont_ids, f)

def update_data():
    last_update = get_last_update_time()
    now = datetime.now()
    if now - last_update < timedelta(days=30):
        print(f"마지막 업데이트 후 30일이 지나지 않았습니다. 다음 업데이트: {last_update + timedelta(days=30)}")
        return

    print(f"데이터 업데이트 시작: {now}")

    scrap_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"
    scrap_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"
    scrap_law_c_url = "https://glaw.scourt.go.kr/wsjo/trty/sjo610.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#//"
    scrap_law_d_url = "https://glaw.scourt.go.kr/wsjo/gchick/sjo300.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p2=01#//"
    scrap_law_j_url = "https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%99%B8%EA%B5%AD%EC%9D%B8%EA%B7%BC%EB%A1%9C%EC%9E%90%EC%9D%98%EA%B3%A0%EC%9A%A9%EB%93%B1%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0"
    crawl_law_e_url = "https://www.gov.kr/portal/foreigner/ko"

    driver = setup_driver()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    index_name = "law_and_visa"

    scraped_cont_ids = load_scraped_cont_ids()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_a = executor.submit(scrap_law_a, driver, scrap_law_a_url, embeddings, index_name, scraped_cont_ids)
            future_b = executor.submit(scrap_law_b, driver, scrap_law_b_url, embeddings, index_name, scraped_cont_ids)
            future_c = executor.submit(scrap_law_b, driver, scrap_law_c_url, embeddings, index_name, scraped_cont_ids)
            future_d = executor.submit(scrap_law_b, driver, scrap_law_d_url, embeddings, index_name, scraped_cont_ids)
            future_j = executor.submit(scrap_law_j, driver, scrap_law_j_url, embeddings, index_name)
            future_e = executor.submit(crawl_law_a, crawl_law_e_url, driver, embeddings, index_name)

            laws_a = future_a.result()
            laws_b = future_b.result()
            laws_c = future_c.result()
            laws_d = future_d.result()
            laws_j = future_j.result()
            crawled_data, total_vectors = future_e.result()

        print(f"스크래핑 및 임베딩 완료: 판례 {len(laws_a)}개")
        print(f"스크래핑 및 임베딩 완료: 법령 {len(laws_b)}개")
        print(f"스크래핑 및 임베딩 완료: 조약 {len(laws_c)}개")
        print(f"스크래핑 및 임베딩 완료: 규칙/예규/선례 {len(laws_d)}개")
        print(f"스크래핑 및 임베딩 완료: 외국인노동법 {len(laws_j)}개")
        print(f"크롤링 및 임베딩 완료: 정부24 외국인 서비스 {len(crawled_data)}개, 총 {total_vectors}개의 벡터")

        # 새로 스크래핑된 cont_id 저장
        for laws in [laws_a, laws_b, laws_c, laws_d]:
            for law in laws:
                if 'cont_id' in law:
                    scraped_cont_ids[law['cont_id']] = True

        save_scraped_cont_ids(scraped_cont_ids)
        set_last_update_time(now)
        print(f"데이터 업데이트 완료: {now}")

    except Exception as e:
        print(f"업데이트 중 오류 발생: {e}")
    finally:
        driver.quit()

def main():
    schedule.every(30).days.do(update_data)
    
    print("데이터 업데이트 스케줄러 시작")
    update_data()  # 초기 실행

    while True:
        schedule.run_pending()
        time.sleep(3600)  # 1시간마다 스케줄 체크

if __name__ == "__main__":
    main()