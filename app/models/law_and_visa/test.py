import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
from openai import OpenAI
import json
import pymilvus
import time

# 환경 변수 로드
load_dotenv()

# Milvus 클라이언트 설정
milvus_client = MilvusClient("http://localhost:19530")

# OpenAI 클라이언트 설정
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(f"pymilvus version: {pymilvus.__version__}")

def check_collection_exists(collection_name):
    try:
        return milvus_client.has_collection(collection_name)
    except Exception as e:
        print(f"Error checking collection existence: {e}")
        return False

def create_collection(collection_name, dim=1536):
    try:
        milvus_client.create_collection(
            collection_name=collection_name,
            fields=[
                {"name": "id", "dtype": "Int64", "is_primary": True},
                {"name": "vector", "dtype": "FloatVector", "dim": dim}
            ]
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")

def create_index(collection_name):
    try:
        milvus_client.create_index(
            collection_name=collection_name,
            field_name="vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        )
        print(f"Index created for collection '{collection_name}'.")
    except Exception as e:
        print(f"Error creating index: {e}")

def load_collection(collection_name):
    try:
        milvus_client.load_collection(collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading collection: {e}")

def print_collection_schema(collection_name):
    try:
        schema = milvus_client.describe_collection(collection_name)
        print("Collection schema:")
        for field in schema['schema']['fields']:
            print(f"Field name: {field['name']}, Type: {field['dtype']}")
    except Exception as e:
        print(f"Error describing collection: {e}")

def get_milvus_data(collection_name, limit=10):
    try:
        results = milvus_client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "vector"],
            limit=limit
        )
        return results
    except Exception as e:
        print(f"Error querying Milvus: {e}")
        return None

def ask_openai(prompt, context):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return None

def main():
    collection_name = "test1"
    
    # 컬렉션 존재 여부 확인
    if not check_collection_exists(collection_name):
        print(f"Collection '{collection_name}' does not exist. Creating...")
        create_collection(collection_name)
        time.sleep(5)  # 생성을 위한 대기 시간
    
    # 인덱스 생성
    create_index(collection_name)
    time.sleep(5)  # 인덱싱을 위한 대기 시간
    
    # 컬렉션 로드
    load_collection(collection_name)
    time.sleep(5)  # 로딩을 위한 대기 시간
    
    # 컬렉션 스키마 출력
    print_collection_schema(collection_name)
    
    # Milvus에서 데이터 가져오기
    milvus_data = get_milvus_data(collection_name)
    if milvus_data is None or len(milvus_data) == 0:
        print("No data available in Milvus. Exiting.")
        return
    
    # 데이터 출력
    print("Milvus에서 가져온 데이터:")
    for item in milvus_data:
        print(json.dumps(item, ensure_ascii=False, indent=2))
    
    # OpenAI에 질문하기
    context = "\n".join([f"ID: {item['id']}, Vector: {item['vector'][:5]}..." for item in milvus_data])
    question = "이 벡터 데이터는 어떤 정보를 나타내나요?"
    
    answer = ask_openai(question, context)
    
    if answer:
        print("\nOpenAI의 답변:")
        print(answer)
    else:
        print("OpenAI로부터 답변을 받지 못했습니다.")

if __name__ == "__main__":
    main()