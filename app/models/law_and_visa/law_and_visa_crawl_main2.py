# app/models/law_and_visa/law_and_visa_crawl_main.py
import sys
import os
import json
from pymilvus import Collection, utility, DataType, connections, FieldSchema, CollectionSchema
from pymilvus.exceptions import MilvusException
from dotenv import load_dotenv
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models.law_and_visa.law_and_visa_util import setup_driver
from app.models.law_and_visa.law_and_visa_crawl import crawl_law_a, crawl_law_b
from app.models.law_and_visa.law_and_visa_embedding import create_embeddings, validate_embeddings, save_embeddings, drop_collection_if_exists, check_collection_schema
from app.models.law_and_visa.law_and_visa_search import search_similar_documents, get_query_embedding

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def main():
    crawl_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"
    crawl_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"
    
    driver = setup_driver()
    
    all_laws = []

    try:
        # 크롤링 데이터 수집
        # laws_a = crawl_law_a(driver, crawl_law_a_url)
        # all_laws.extend(laws_a)
        
        # laws_b = crawl_law_b(driver, crawl_law_b_url)
        # all_laws.extend(laws_b)
        
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
    print(f"첫 번째 법률 정보 구조: {all_laws[0].keys()}")

    if all_laws:
        print("임베딩 처리 시작")
        setup_milvus()
        collection_name = "law_embeddings"
        dim = 1536  # text-embedding-ada-002 모델의 임베딩 차원

        # 컬렉션 존재 여부 확인 및 생성
        if not utility.has_collection(collection_name):
            print(f"컬렉션 '{collection_name}' 생성 중")
            create_collection(collection_name, dim)
        # drop_collection_if_exists("test")
        check_collection_schema(collection_name)
        collection = Collection(name=collection_name)

        # 임베딩 저장
        embeddings = create_embeddings(all_laws)
        validate_embeddings(embeddings, dim)

        # 'laws' 인자를 추가하여 save_embeddings 호출
        save_embeddings(collection, embeddings, all_laws)

        # 컬렉션 로드
        print(f"컬렉션 '{collection_name}'을(를) 메모리에 로드합니다.")
        try:
            collection.load()
        except MilvusException as e:
            print(f"컬렉션 로드 중 오류 발생: {e}")
            return

        # 검색 기능 테스트
        query_text = "외국인"
        query_embedding = get_query_embedding(query_text)
        results = search_similar_documents(collection, query_embedding)

        print(f'\n"{query_text}"와 유사한 상위 5개 문서:')
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 제목: {result.entity.get('title')}")
            print(f"   유사도 점수: {result.score:.4f}")
            print(f"   내용 일부: {result.entity.get('content')[:200]}...")

        # OpenAI에게 질문과 벡터 데이터베이스 결과 전송
        response = get_openai_response(query_text, results)
        print(f"OpenAI 응답: {response}")

    else:
        print("크롤링된 데이터가 없습니다.")

    print("전체 프로세스 완료")

def setup_milvus():
    connections.connect("default", host="localhost", port="19530")

def create_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Law Embeddings Collection")
    collection = Collection(name=collection_name, schema=schema)
    return collection

def get_openai_response(query, documents):
    context = "\n".join([doc.entity.get("content") for doc in documents])
    prompt = f"질문: {query}\n\n관련 문서:\n{context}\n\n답변:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    main()
