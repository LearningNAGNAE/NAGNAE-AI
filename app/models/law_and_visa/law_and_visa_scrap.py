import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models.law_and_visa.law_and_visa_util import setup_driver
from app.models.law_and_visa.law_and_visa_embedding import scrap_law_a, scrap_law_b, scrap_law_j

load_dotenv()


def main():
    #a.종합법률정보: 판례
    scrap_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"

    #b.종합법률정보: 법령
    # scrap_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"
    scrap_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#1722409948531"

    #c.종합법률정보: 조약
    scrap_law_c_url = "https://glaw.scourt.go.kr/wsjo/trty/sjo610.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#//"

    #d.종합법률정보: 규칙/예규/선례
    scrap_law_d_url = "https://glaw.scourt.go.kr/wsjo/gchick/sjo300.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p2=01#//"

    #j. 국가법령정보센터: 외국인고용법
    scrap_law_j_url = "https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%99%B8%EA%B5%AD%EC%9D%B8%EA%B7%BC%EB%A1%9C%EC%9E%90%EC%9D%98%EA%B3%A0%EC%9A%A9%EB%93%B1%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0"
    
    driver = setup_driver()

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        # index_name = "law_and_visa"
        index_name = "test"

        # ThreadPoolExecutor 블록: 웹 스크래핑 및 임베딩 생성
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # future_a = executor.submit(scrap_law_a, driver, scrap_law_a_url, embeddings, index_name) #완료
            future_b = executor.submit(scrap_law_b, driver, scrap_law_b_url, embeddings, index_name)
            # future_c = executor.submit(scrap_law_b, driver, scrap_law_c_url, embeddings, index_name) 
            # future_d = executor.submit(scrap_law_b, driver, scrap_law_d_url, embeddings, index_name) 
            # future_j = executor.submit(scrap_law_j, driver, scrap_law_j_url, embeddings, index_name) #완료

            # laws_a = future_a.result()
            laws_b = future_b.result()
            # laws_c = future_c.result()
            # laws_d = future_d.result()
            # laws_j = future_j.result()
            # print("")
        # 결과 출력
        # print(f"스크래핑 및 임베딩 완료: 판례 {len(laws_a)}개")
        print(f"스크래핑 및 임베딩 완료: 법령 {len(laws_b)}개")
        # print(f"스크래핑 및 임베딩 완료: 조약 {len(laws_c)}개")
        # print(f"스크래핑 및 임베딩 완료: 규칙/예규/선례 {len(laws_d)}개")
        # print(f"스크래핑 및 임베딩 완료: 외국인노동법 {len(laws_j)}개")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        driver.quit()

    print("전체 프로세스 완료")

if __name__ == "__main__":
    main()