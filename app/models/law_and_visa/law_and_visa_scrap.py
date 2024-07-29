import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models.law_and_visa.law_and_visa_util import setup_driver
from app.models.law_and_visa.law_and_visa_search import scrap_law_a, scrap_law_b

load_dotenv()

def main():
    # # a. 종합법률정보: 판례
    scrap_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"

    # # b. 종합법률정보: 법령
    scrap_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"

    # # c. 종합법률정보: 조약
    scrap_law_c_url = "https://glaw.scourt.go.kr/wsjo/trty/sjo610.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#//"

    # d. 종합법률정보: 규칙/예규/선례
    scrap_law_d_url = "https://glaw.scourt.go.kr/wsjo/gchick/sjo300.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p2=01#//"

    driver = setup_driver()

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # future_a = executor.submit(scrap_law_a(driver, scrap_law_a_url, embeddings, "law"))
            future_b = executor.submit(scrap_law_b(driver, scrap_law_b_url, embeddings, "law"))
            # future_c = executor.submit(scrap_law_b(driver, scrap_law_c_url, embeddings, "law"))
            # future_d = executor.submit(scrap_law_b(driver, scrap_law_d_url, embeddings, "law"))

            # count_a = future_a.result()
            count_b = future_b.result()
            # count_c = future_c.result()
            # count_d = future_d.result()

        # print(f"임베딩 완료: 판례 {count_a}개")
        print(f"임베딩 완료: 법령 {count_b}개")
        # print(f"임베딩 완료: 조약 {count_c}개")
        # print(f"임베딩 완료: 규칙/예규/선례 {count_d}개")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        driver.quit()

    print("전체 프로세스 완료")

if __name__ == "__main__":
    main()