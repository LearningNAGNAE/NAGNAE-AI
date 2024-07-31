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

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI()

# Define input data model
class Query(BaseModel):
    question: str
    session_id: str

embeddings = OpenAIEmbeddings()
index_name = "faiss_index_law_and_visa_page"
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, index_name)

# Load FAISS index
if not os.path.exists(index_path):
    raise HTTPException(status_code=500, detail=f"FAISS index not found: {index_path}")

vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

print(f"FAISS index size: {vector_store.index.ntotal}")

# Use vector store as retriever
retriever = vector_store.as_retriever()

# Define system prompt
system_prompt = """
# AI Assistant for Korean Law, Visa, Labor Rights, and General Information

## Role and Responsibility
You are a specialized AI assistant providing information on Korean law, visa, labor rights, and general topics for both foreigners and citizens in Korea. Your goals are to:

1. Explain information kindly and clearly in the language of the query.
2. Provide accurate and easy-to-understand information.
3. Offer both general and specific information when relevant.
4. Provide guidance on labor rights and workplace issues.

## Guidelines

1. Language: Respond in the same language as the question (e.g., Korean for Korean questions, Chinese for Chinese questions).
2. Information Scope: Provide information on laws, visas, labor rights, and general topics related to living and working in Korea.
3. Visa Information: Clearly distinguish between general visa rules and specific visa type regulations.
4. Labor Rights: Offer information on worker's rights, including wage payment, working hours, and dispute resolution processes.
5. Uncertainty: If uncertain, respond with: 
   - Korean: "제공된 정보로는 정확한 답변을 드릴 수 없습니다. 구체적인 조언을 위해 관련 정부 기관이나 노동법 전문가에게 문의하시는 것이 좋겠습니다."
   - Chinese: "根据提供的信息，我无法给出准确的答复。建议您向相关政府机构或劳动法专家咨询以获取具体建议。"
   - English: "I cannot provide an accurate answer with the given information. I recommend consulting the relevant government office or a labor law expert for specific advice."
6. Legal Disclaimers: Emphasize that laws may change and encourage verifying current regulations.
7. Objectivity: Stick to factual information without personal opinions.
8. Cultural Sensitivity: Be aware of cultural differences and provide context when necessary.
9. Resources: When applicable, provide information about relevant government agencies, legal aid organizations, or official documentation for further assistance.

Always approach each query systematically to ensure accurate, helpful, and responsible assistance. For labor-related issues, provide general guidance on worker's rights and suggest official channels for seeking help or filing complaints.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up prompt template
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

# Initialize ChatOpenAI model
model = ChatOpenAI(temperature=0)

# Configure retrieval chain
retrieval_chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    }
    | chat_prompt
    | model
    | StrOutputParser()
)

# Memory storage
memory_store = {}

def get_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# Set up RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Language detection function
def detect_language(text):
    system_prompt = "You are a language detection expert. Detect the language of the given text."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = chat.invoke(messages)
    return response.content.strip()

# Request validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )



import re 
def format_text_for_web(text):
    try:
        # 단락 나누기: 빈 줄을 기준으로 단락을 나눕니다.
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 각 단락을 <p> 태그로 감싸고, 줄바꿈을 <br> 태그로 변환합니다.
        formatted_paragraphs = ['<p>' + p.replace('\n', '<br>') + '</p>' for p in paragraphs]
        
        # 모든 단락을 하나의 문자열로 결합합니다.
        return ''.join(formatted_paragraphs)
    except Exception as e:
        logging.error(f"Error in format_text_for_web: {str(e)}")
        return f"<p>{text}</p>"  # 오류 발생 시 기본 형식으로 반환

# CORS 미들웨어 추가(임시)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/law")
async def chat_endpoint(query: Query):
    try:
        question = query.question if isinstance(query.question, str) else str(query.question)
        session_id = query.session_id
        print(f"Question: {question}")

        # 언어 감지
        language = detect_language(question)

        if retriever is None:
            return {
                "question": question,
                "answer": format_text_for_web("죄송합니다. 현재 시스템에 문제가 있어 답변을 드릴 수 없습니다. 나중에 다시 시도해 주세요."),
                "detected_language": language,
            }

        # RunnableWithMessageHistory를 사용하여 응답 생성
        response = chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

        # 응답 텍스트를 웹 표시에 적합하게 포맷팅
        formatted_response = format_text_for_web(response)

        return {
            "question": question,
            "answer": formatted_response,
            "detected_language": language,
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        error_message = "죄송합니다. 요청을 처리하는 동안 오류가 발생했습니다. 나중에 다시 시도해 주세요."
        return {
            "question": question,
            "answer": format_text_for_web(error_message),
            "detected_language": detect_language(question),
        }

# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)