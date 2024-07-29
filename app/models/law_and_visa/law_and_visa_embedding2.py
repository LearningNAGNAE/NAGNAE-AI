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
    
    # 컬렉션 생성
    collection = Collection(name=collection_name, schema=schema)
    return collection

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
