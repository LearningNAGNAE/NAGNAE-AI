import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 환경 변수 로드
load_dotenv()

# FastAPI 애플리케이션 생성
app = FastAPI()

# 입력 데이터 모델 정의
class Query(BaseModel):
    question: str

# 모델 초기화
embeddings = OpenAIEmbeddings()
index_name = "faiss_index_law_page"
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, index_name)

# FAISS 인덱스 로드
if not os.path.exists(index_path):
    raise HTTPException(status_code=500, detail=f"FAISS 인덱스를 찾을 수 없습니다: {index_path}")

vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# 시스템 프롬프트 정의
system_prompt = """
당신은 한국의 외국인 노동자와 유학생을 위한 법률 상담 전문가이자 통역/번역가입니다. 
주어진 정보를 바탕으로 친절하고 명확하게 설명해주세요. 
법률 정보를 제공할 때는 정확성을 유지하면서도 이해하기 쉽게 설명해야 합니다.
답변할 수 없는 질문에 대해서는 솔직히 모른다고 인정하고, 필요한 경우 전문가와 상담을 권유하세요.
"""

# 프롬프트 템플릿 설정
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_template = """
관련 정보: {context}

질문: {question}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
ai_message_prompt = AIMessagePromptTemplate.from_template("답변:")
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
    ai_message_prompt
])

# 대화 기록을 유지하기 위한 메모리
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# 언어 감지 함수
def detect_language(text):
    system_prompt = "You are a language detection expert. Detect the language of the given text."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = chat.invoke(messages)
    return response.content.strip()

# 요청 검증 예외 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

# API 엔드포인트 정의
@app.post("/ask")
async def ask_question(query: Query):
    try:
        # 질문이 문자열인지 확인
        question = query.question if isinstance(query.question, str) else str(query.question)

        # 언어 감지
        language = detect_language(question)

        # FAISS를 사용한 유사도 검색
        docs = vector_store.similarity_search(question, k=4)
        context = "\n".join([doc.page_content for doc in docs])

        # 프롬프트 형식화
        formatted_prompt = chat_prompt.format_prompt(context=context, question=question)

        # LLM에 요청
        response = qa_chain({"question": question, "chat_history": []})

        # 응답 반환
        return {
            "question": question,
            "answer": response['answer'],
            "detected_language": language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# 메인 실행 부분
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)