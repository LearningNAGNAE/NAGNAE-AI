import openai
from dotenv import load_dotenv
import os
from pymilvus import connections, Collection
from app.models.law_and_visa.law_and_visa_embedding import get_query_embedding
from app.models.law_and_visa.law_and_visa_search import search_similar_documents

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

def setup_milvus():
    connections.connect("default", host="localhost", port="19530")

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def generate_prompt(question, context):
    return f"""
당신은 한국 법률 전문가입니다. 주어진 컨텍스트를 바탕으로 다음 질문에 대해 정확하고 간결하게 답변해주세요.
필요한 경우 관련 법조항을 인용하세요.

컨텍스트:
{context}

질문: {question}

답변:
"""

def answer_legal_question(question):
    setup_milvus()
    collection_name = "law_embeddings"
    collection = Collection(collection_name)
    collection.load()

    # 질문에 대한 임베딩 생성
    query_embedding = get_query_embedding(question)
    
    # Milvus에서 유사한 문서 검색
    results = search_similar_documents(collection, query_embedding, top_k=3)
    
    # 컨텍스트 생성
    context = "\n\n".join([f"제목: {result.entity.get('title')}\n내용: {result.entity.get('content')[:500]}..." for result in results])
    
    # 프롬프트 생성 및 OpenAI API 호출
    prompt = generate_prompt(question, context)
    answer = get_completion(prompt)
    
    return answer

# 사용 예
if __name__ == "__main__":
    question = "외국인의 국내 취업에 관한 기본적인 법적 요건은 무엇인가요?"
    answer = answer_legal_question(question)
    print(f"질문: {question}\n\n답변: {answer}")