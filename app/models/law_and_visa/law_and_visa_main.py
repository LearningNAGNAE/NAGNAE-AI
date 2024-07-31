import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseChatMessageHistory
import logging
import traceback

# 환경 변수 로드
load_dotenv()

# FastAPI 애플리케이션 생성
app = FastAPI()

# 입력 데이터 모델 정의
class Query(BaseModel):
    question: str
    session_id: str

embeddings = OpenAIEmbeddings()
index_name = "faiss_index_law_and_visa_page"
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, index_name)



# FAISS 인덱스 로드
if not os.path.exists(index_path):
    raise HTTPException(status_code=500, detail=f"FAISS 인덱스를 찾을 수 없습니다: {index_path}")

vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

print(f"FAISS 인덱스 크기: {vector_store.index.ntotal}")



# 벡터 저장소를 검색기로 사용합니다.
retriever = vector_store.as_retriever()



# ------------------ 시스템 프롬프트 정의 -----------------------
system_prompt = """
# AI Assistant for Korean Law and Visa Information

## Role and Responsibility
You are a specialized AI assistant providing law and visa-related information for foreigners residing in Korea. Your primary goals are to:

1. Explain information kindly and clearly
2. Maintain accuracy while being easy to understand
3. Adhere strictly to the given guidelines

## Step-by-step Guidelines

### Step 1: Information Source
- Use **ONLY** the content from the given context information to answer
- Do NOT include any information that is not in the context

### Step 2: Relevance
- Exclude information irrelevant to the question
- Focus on providing a direct and pertinent answer

### Step 3: Clarity and Conciseness
- Write your answers concisely and clearly
- Use simple language where possible, avoiding jargon unless necessary

### Step 4: Handling Uncertainty
- If uncertain or unable to answer with the given information, respond with:
  > "I cannot provide an accurate answer with the given information. I recommend consulting a legal professional or the appropriate government office for specific advice."

### Step 5: Legal Disclaimers
- When discussing legal matters:
  1. Emphasize that laws may change
  2. Encourage users to verify current regulations

### Step 6: Visa-related Queries
- For visa-related questions, always advise:
  > "Please check with the Korean Immigration Service for the most up-to-date information."

### Step 7: Objectivity
- Avoid giving personal opinions or interpretations of the law
- Stick to factual information provided in the context

## Final Reminder
Always approach each query systematically, following these steps to ensure accurate, helpful, and responsible assistance.
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------

# 프롬프트 템플릿 설정
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_template = """
context: {context}

question: {question}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
ai_message_prompt = AIMessagePromptTemplate.from_template("answer:")
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
    ai_message_prompt
])

# ChatOpenAI 모델을 초기화합니다.
model = ChatOpenAI(temperature=0)

# 검색 체인을 구성합니다.
retrieval_chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    }
    | chat_prompt
    | model
    | StrOutputParser()
)


# 메모리 저장소
memory_store = {}

def get_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# RunnableWithMessageHistory 설정
chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
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



# --------------------- API 엔드포인트 정의----------------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(query: Query):
    try:
        question = query.question if isinstance(query.question, str) else str(query.question)
        session_id = query.session_id
        print(f"질문: {question}")

        # 언어 감지
        language = detect_language(question)

        # RunnableWithMessageHistory를 사용하여 응답 생성
        response = chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

        return {
            "question": question,
            "answer": response,
            "detected_language": language,
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# 메인 실행 부분
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)