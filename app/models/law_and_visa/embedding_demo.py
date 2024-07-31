# app/models/law_and_visa/law_and_visa_embedding.py
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from pymilvus import Collection, utility, DataType, connections, CollectionSchema, FieldSchema

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI API를 사용하여 임베딩 생성
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 임베딩 생성
def create_embeddings(laws):
    embeddings = []
    for i, law in enumerate(laws):
        embedding = get_embedding(law['content'])
        embeddings.append({'id': i, 'vector': embedding})  # 'i'를 ID로 사용
    return embeddings

def validate_embeddings(embeddings, dim):
    for emb in embeddings:
        if not isinstance(emb["vector"], list) or len(emb["vector"]) != dim:
            raise ValueError(f"임베딩 데이터가 리스트 형식이 아니거나 {dim} 차원이 아닙니다.")
        if not all(isinstance(x, float) for x in emb["vector"]):
            raise ValueError("임베딩 데이터에 부동소수점(float) 타입이 아닌 값이 포함되어 있습니다.")

# 컬렉션 생성
def create_collection(collection_name, dim):
    # 1. Set up a Milvus client
    connections.connect(alias="default", host="localhost", port="19530")

    # 2. Check if the collection exists and drop it if it does
    if utility.has_collection(collection_name):
        print(f"컬렉션 '{collection_name}'이(가) 이미 존재하므로 삭제합니다.")
        utility.drop_collection(collection_name)

    # 3. Create schema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id_field, vector_field], auto_id=False)

    # 4. Create collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"컬렉션 '{collection_name}'이(가) 생성되었습니다.")

def create_index(collection_name):
    collection = Collection(name=collection_name)
    # 인덱스 생성 (예: IVF_FLAT)
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
        "metric_type": "L2"
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"인덱스가 컬렉션 '{collection_name}'에 생성되었습니다.")

def save_embeddings(collection, embeddings):
    ids = [emb["id"] for emb in embeddings]
    vectors = [emb["vector"] for emb in embeddings]
    # Insert data into the collection
    res = collection.insert([ids, vectors])
    print("임베딩 저장 완료:", res)

def drop_collection_if_exists(collection_name):
    if utility.has_collection(collection_name):
        print(f"컬렉션 '{collection_name}'이(가) 이미 존재하므로 삭제합니다.")
        utility.drop_collection(collection_name)

def load_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.load()
    print(f"컬렉션 '{collection_name}'이(가) 메모리에 로드되었습니다.")

def check_collection_schema(collection_name):
    collection = Collection(name=collection_name)
    schema = collection.schema
    print("컬렉션 스키마:")
    for field in schema.fields:
        print(f"Field name: {field.name}, Type: {field.dtype}, Dim: {field.dim}")

def search_similar_documents(collection, query_embedding):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param=search_params,
        limit=5
    )
    return results[0]

def main():
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
    embeddings = create_embeddings(all_laws)
    validate_embeddings(embeddings, dim)
    
    collection = Collection(name=collection_name)
    save_embeddings(collection, embeddings)
    
    # 컬렉션 로드
    load_collection(collection_name)
    
    # 검색 기능 테스트
    query_text = "외국인"
    query_embedding = get_embedding(query_text)
    results = search_similar_documents(collection, query_embedding)
    
    print(f'\n"{query_text}"와 유사한 상위 5개 문서:')
    for i, result in enumerate(results, 1):
        print(f"   유사도 점수: {result.score:.4f}")

if __name__ == "__main__":
    main()
