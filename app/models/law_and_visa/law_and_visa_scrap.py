import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models.law_and_visa.law_and_visa_util import setup_driver
from app.models.law_and_visa.law_and_visa_crawl import scrap_law_a, scrap_law_b

# 환경 변수 로드
load_dotenv()

def main():
    #a.종합법률정보: 판례
    scrap_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"

    #b.종합법률정보: 법령
    scrap_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"

    #c.종합법률정보: 조약
    scrap_law_c_url = "https://glaw.scourt.go.kr/wsjo/trty/sjo610.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#//"

    #d.종합법률정보: 규칙/예규/선례
    scrap_law_d_url = "https://glaw.scourt.go.kr/wsjo/gchick/sjo300.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p2=01#//"


    driver = setup_driver()
    all_laws = []

    try:
        # 크롤링 데이터 수집
        #a.종합법률정보: 판례
        # laws_a = scrap_law_a(driver, scrap_law_a_url)
        # all_laws.extend(laws_a)

        #b.종합법률정보: 법령
        # laws_b = scrap_law_b(driver, scrap_law_b_url)
        # all_laws.extend(laws_b)

        #c.종합법률정보: 조약
        # laws_c = scrap_law_b(driver, scrap_law_c_url)
        # all_laws.extend(laws_c)

        #d.종합법률정보: 규칙/예규/선례
        laws_d = scrap_law_b(driver, scrap_law_d_url)
        all_laws.extend(laws_d)


        # 테스트용 데이터 추가
        test_law = {
            "title": "외국인 체포 및 구속 시 영사통보권에 관한 판례",
            "content": """
            【판시사항】
            [1] 영사관계에 관한 비엔나협약 제36조 제1항 (b)호, 경찰수사규칙 제91조 제2항, 제3항에서 외국인을 체포·구속하는 경우 지체 없이 외국인에게 영사통보권 등이 있음을 고지하고, 외국인의 요청이 있는 경우 영사기관에 체포·구금 사실을 통보하도록 정한 취지 / 수사기관이 외국인을 체포하거나 구속하면서 지체 없이 영사통보권 등이 있음을 고지하지 않은 경우, 체포나 구속 절차가 위법한지 여부(적극)
            ... (생략) ...
            """
        }
        all_laws.append(test_law)
    finally:
        driver.quit()

    print("크롤링 완료")
    print(f"수집된 법률 정보 개수: {len(all_laws)}")

 
    if all_laws:
        print("임베딩 처리 시작")

        # 문서 준비
        documents = []
        for law in all_laws:
            content = f"제목: {law['title']}\n\n"
            if isinstance(law['content'], dict):
                for section, section_content in law['content'].items():
                    content += f"{section}:\n{section_content}\n\n"
            else:
                content += f"내용:\n{law['content']}\n\n"
            documents.append(Document(page_content=content, metadata={"title": law['title']}))
            
        # 텍스트 스플리터 설정
        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 1500,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_documents(documents)

        # OpenAI 임베딩 설정
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        try:
            # Milvus 연결 설정
            connections.connect(host="localhost", port="19530")

            collection_name = "law_and_visa"

            # 컬렉션이 이미 존재하는 경우 삭제
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)

            # Milvus 스키마 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)  # OpenAI의 임베딩 차원
            ]
            schema = CollectionSchema(fields, "Law collection")

            # 컬렉션 생성
            collection = Collection(name=collection_name, schema=schema)

            # 인덱스 생성
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="vector", index_params=index_params)

            # Milvus 벡터 저장소 설정
            vector_store = Milvus.from_documents(
                texts,
                embeddings,
                collection_name=collection_name,
                connection_args={"host": "localhost", "port": "19530"}
            )

            print("임베딩 저장 완료")

        except Exception as e:
            print(f"Milvus 에러 발생: {e}")
            print("벡터 저장소 생성에 실패하여 프로그램을 종료합니다.")
            return

    else:
        print("크롤링된 데이터가 없습니다.")

    print("전체 프로세스 완료")

if __name__ == "__main__":
    main()