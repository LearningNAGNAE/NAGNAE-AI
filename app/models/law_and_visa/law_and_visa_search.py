import openai
from dotenv import load_dotenv
import os
from pymilvus import connections, Collection

# OpenAI API 키 설정
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Milvus 연결 설정
def setup_milvus():
    connections.connect("default", host="localhost", port="19530")

# 쿼리 텍스트의 임베딩 생성
def get_query_embedding(query_text):
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Milvus에서 유사한 문서 검색
def search_similar_documents(collection, query_embedding, top_k=5):
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "content"]
    )
    return results[0]

# 메인 검색 함수
def main():
    setup_milvus()
    collection_name = "law_embeddings"
    collection = Collection(collection_name)
    collection.load()

    query_text = "외국인"
    query_embedding = get_query_embedding(query_text)
    
    results = search_similar_documents(collection, query_embedding)
    
    print(f'"{query_text}"와 유사한 상위 5개 문서:')
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 제목: {result.entity.get('title')}")
        print(f"   유사도 점수: {result.score:.4f}")
        print(f"   내용 일부: {result.entity.get('content')[:200]}...")

if __name__ == "__main__":
    main()