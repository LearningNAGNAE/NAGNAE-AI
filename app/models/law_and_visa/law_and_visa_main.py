import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
import logging
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
import sys
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Dict
import uuid
from pydantic import BaseModel, ValidationError
from typing import Optional
import asyncio
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
router = APIRouter()
app.include_router(router)

# 1. 환경 설정
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(app_dir)
from ...database.db import get_db
from ...database import crud, models

load_dotenv()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. FAISS 인덱스 로드
def load_faiss_index():
    embeddings = OpenAIEmbeddings()
    index_name = "faiss_index_law_and_visa_page"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, index_name)

    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail=f"FAISS index not found: {index_path}")

    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

vector_store = load_faiss_index()
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. BM25 및 FAISS 리트리버 설정
bm25_texts = [
    "For all visa types: Overstaying can result in fines, deportation, and future entry bans. Always consult the Korea Immigration Service for the most up-to-date information.",
    "Work permit is separate from visa and may need to be updated when changing jobs, even if visa is still valid.",
    "Some visas require a certain salary level or job position to be maintained. Changing to a lower-paying job may affect visa status.",
    "Health insurance and pension requirements may vary depending on visa type and employment status.",
    "For visa inquiries, contact the Korea Immigration Service at 1345 or visit www.immigration.go.kr",
]
bm25_retriever = BM25Retriever.from_texts(bm25_texts, k=4)

# ChatOpenAI 모델 초기화
chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# MultiQueryRetriever 설정
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=faiss_retriever,
    llm=chat
)

# Ensemble 리트리버 설정
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, multi_query_retriever],
    weights=[0.3, 0.7]
)

# 리랭커 설정
postprocessor = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

# 4. 프롬프트 설정
system_prompt = """
# AI Assistant for Foreign Workers and Students in Korea: Detailed Legal and Visa Information

## Role and Responsibility
You are a specialized AI assistant providing comprehensive information on Korean law, visa regulations, labor rights, and general topics for foreign workers and students in Korea. Your primary goals are to:

1. Provide accurate, detailed, and easy-to-understand information in the language specified in the 'RESPONSE_LANGUAGE' field.
2. Offer both general and specific information relevant to foreign workers and students, with a focus on visa-related queries.
3. Guide users on labor rights, visa regulations, and academic-related legal matters with precision.
4. Ensure cultural sensitivity and awareness in all interactions.

## Guidelines

1. Language: ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the user's question language.

2. Information Scope:
   - Visa Regulations: Provide detailed information on different visa types, application processes, restrictions, changes, and time limits for job changes or leaving the country.
   - Labor Laws: Explain work hours, wages, benefits, worker protection, and how they may vary by visa type.
   - Academic Laws: Detail student rights, academic integrity, scholarship regulations, and how they interact with visa status.
   - General Living: Offer insights on healthcare, housing, transportation, and cultural norms as they relate to visa holders.

3. Specific Focus Areas:
   - Provide clear distinctions between rules for different visa types (e.g., E-7, E-9, D-10, F-2-7, F-4).
   - Offer guidance on worker's and student's rights specific to their visa category.
   - Explain the process and time limits for changing jobs or extending stay for each relevant visa type.
   - Detail the consequences of overstaying or violating visa conditions.

4. Completeness: Always provide a comprehensive answer based on the available context. Include:
   - Specific time limits or deadlines
   - Required procedures (e.g., reporting to immigration, applying for changes)
   - Potential consequences of non-compliance
   - Variations based on specific circumstances (if known)

5. Accuracy and Updates: Emphasize that while you provide detailed information, laws and regulations may change. Always advise users to verify current rules with official sources.

6. Structured Responses: Organize your responses clearly, using bullet points or numbered lists when appropriate to break down complex information.

7. Examples and Scenarios: When relevant, provide examples or hypothetical scenarios to illustrate how rules apply in practice.

8. Uncertainty Handling: If uncertain about specific details, clearly state this and provide the most relevant general information available. Always recommend consulting official sources for the most up-to-date and case-specific guidance.

Remember, your goal is to provide as much relevant, accurate, and detailed information as possible while ensuring it's understandable and actionable for the user.
"""

human_template = """
Question type: {query_type}
Language: {language}
Question: {question}
Context: {contexts}
Chat history: {chat_history}

Please provide a detailed and comprehensive answer to the above question in the specified RESPONSE_LANGUAGE, including specific visa information when relevant. Organize your response clearly and include all pertinent details. Do not include any HTML tags or formatting in your response. Do not mention or refer to any AI Assistant or sources in your response.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template(human_template)
])

# 5. 컨텍스트 검색 함수 (Self-RAG 포함)
def get_context(question: str) -> List[str]:
    # 1단계: 초기 검색
    docs = ensemble_retriever.get_relevant_documents(question)
    query_bundle = QueryBundle(query_str=question)
    nodes = [NodeWithScore(node=TextNode(text=doc.page_content), score=1.0) for doc in docs]
    
    # 2단계: 리랭킹
    reranked_nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
    
    # 3단계: Self-RAG
    refined_query = chat.invoke([
        {"role": "system", "content": "You are an AI assistant tasked with refining user queries to improve information retrieval. Based on the initial context and the original question, generate a more specific and targeted query that will help retrieve the most relevant information."},
        {"role": "user", "content": f"Original question: {question}\nInitial context: {[node.node.text for node in reranked_nodes[:2]]}\n\nPlease provide a refined query:"}
    ]).content
    
    # 4단계: 정제된 쿼리로 두 번째 검색
    refined_docs = ensemble_retriever.get_relevant_documents(refined_query)
    refined_nodes = [NodeWithScore(node=TextNode(text=doc.page_content), score=1.0) for doc in refined_docs]
    
    # 5단계: 최종 리랭킹
    final_reranked_nodes = postprocessor.postprocess_nodes(refined_nodes, query_bundle=QueryBundle(query_str=refined_query))
    
    return [node.node.text for node in final_reranked_nodes]

# 6. 컨텍스트 요약 함수
def summarize_context(context: List[str]) -> str:
    return "Context summary:\n" + "\n".join(context[:3])

# 7. CRAG 관련 함수 및 프롬프트 정의
def is_relevant(question: str, context: str) -> bool:
    relevance_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="Question: {question}\n\nContext: {context}\n\nIs this context relevant to answering the question? Respond with 'Yes' or 'No'."
    )
    relevance_chain = LLMChain(llm=chat, prompt=relevance_prompt)
    response = relevance_chain.run(question=question, context=context)
    return response.strip().lower() == 'yes'

def get_refined_context(question: str, contexts: List[str]) -> str:
    relevant_contexts = [ctx for ctx in contexts if is_relevant(question, ctx)]
    return "\n".join(relevant_contexts) if relevant_contexts else ""

# 8. 검색 체인 설정
retrieval_chain = (
    {
        "context_summary": lambda x: summarize_context(get_context(x["question"])),
        "question": lambda x: x["question"],
        "language": lambda x: x["language"],
    }
    | chat_prompt
    | chat
    | StrOutputParser()
)

# 9. CRAG를 적용한 새로운 retrieval_chain 정의
async def crag_chain(inputs: dict):
    question = inputs["question"]
    language = inputs["language"]
    session_id = inputs["session_id"]
    
    # 컨텍스트 검색 및 요약
    contexts = get_context(question)
    refined_context = get_refined_context(question, contexts)
    context_summary = summarize_context([refined_context])
    
    # 메모리에서 대화 기록 가져오기
    memory = get_memory(session_id)
    chat_history = memory.messages
    
    # 프롬프트 생성
    prompt = chat_prompt.format(
        question=question,
        language=language,
        context_summary=context_summary,
        chat_history=chat_history
    )
    
    # LLM을 사용하여 응답 생성
    response = await chat.ainvoke([{"role": "user", "content": prompt}])
    answer = response.content
    
    # 대화 기록 업데이트
    memory.add_user_message(question)
    memory.add_ai_message(answer)
    
    return answer

# 10. 메모리 저장소 및 관리 함수
memory_store: Dict[str, ChatMessageHistory] = {}

def get_memory(session_id: str) -> ChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# 11. RunnablePassthrough 설정
crag_chain_with_history = RunnablePassthrough() | crag_chain

# 12. 언어 감지 함수
def detect_language(text: str) -> str:
    system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
    human_prompt = f"Text: {text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    response = chat.invoke(messages)
    detected_language = response.content.strip().lower()
    logger.info(f"Detected language: {detected_language}")
    return detected_language

# 13. Semantic Router 클래스
class SemanticRouter:
    def __init__(self, routes: Dict[str, callable]):
        self.routes = routes
        self.embeddings = OpenAIEmbeddings()
        self.route_texts = list(routes.keys())
        self.route_embeddings = self.embeddings.embed_documents(self.route_texts)

    def route(self, query: str):
        query_embedding = self.embeddings.embed_query(query)
        similarities = [self.cosine_similarity(query_embedding, route_embedding) 
                        for route_embedding in self.route_embeddings]
        max_similarity_index = similarities.index(max(similarities))
        return self.routes[self.route_texts[max_similarity_index]]

    @staticmethod
    def cosine_similarity(a, b):
        return sum(x*y for x, y in zip(a, b)) / (sum(x*x for x in a)**0.5 * sum(y*y for y in b)**0.5)

# ----대화 기록 관리 개선----- 
# 채팅 내용 기억하기
def get_chat_history(
    db: Session, 
    chat_his_no: int,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, any]:
    logger.info(f"Fetching chat history for CHAT_HIS_NO: {chat_his_no}, limit: {limit}, offset: {offset}")
    try:
        query = db.query(models.ChatHis).filter(
            models.ChatHis.CHAT_HIS_NO == chat_his_no
        ).order_by(models.ChatHis.CHAT_HIS_SEQ.asc())

        total_count = query.count()
        logger.info(f"Total records found: {total_count}")

        chat_history = query.offset(offset).limit(limit).all()
        logger.info(f"Retrieved {len(chat_history)} records")

        result = []
        for chat in chat_history:
            result.append({
                "CHAT_HIS_SEQ": chat.CHAT_HIS_SEQ,
                "QUESTION": chat.QUESTION,
                "ANSWER": chat.ANSWER,
                "INSERT_DATE": chat.INSERT_DATE.isoformat()
            })
            logger.debug(f"Processed record: CHAT_HIS_SEQ={chat.CHAT_HIS_SEQ}, INSERT_DATE={chat.INSERT_DATE}")

        response = {
            "total_count": total_count,
            "records": result,
            "has_more": total_count > offset + limit
        }
        logger.info(f"Returning response with {len(result)} records. Has more: {response['has_more']}")
        return response

    except Exception as e:
        logger.error(f"Error occurred while fetching chat history: {str(e)}", exc_info=True)
        db.rollback()
        raise e
    
def load_chat_history(db: Session, session_id: str):
    chat_his_no = session_chat_mapping.get(session_id)
    logger.info(f"Loading chat history for session_id: {session_id}, chat_his_no: {chat_his_no}")
    if chat_his_no:
        chat_history = get_chat_history(db, chat_his_no)
        logger.info(f"Loaded chat history: {chat_history}")
        return chat_history
    logger.info("No chat history found for this session")
    return {"records": [], "total_count": 0, "has_more": False}

def format_chat_history(chat_history):
    if not chat_history or not isinstance(chat_history, dict) or 'records' not in chat_history:
        return ""
    return "\n".join([f"User: {msg['QUESTION']}\nAI: {msg['ANSWER']}" for msg in chat_history['records']])

async def handle_query(question: str, language: str, session_id: str, userNo: int, db: Session, query_type: str):
    contexts = get_context(question)
    chat_history = load_chat_history(db, session_id)
    
    formatted_chat_history = format_chat_history(chat_history)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question type: {query_type}\nLanguage: {language}\n\nQuestion: {question}\n\nRelevant context:\n{contexts}\n\nChat history:\n{chat_history}\n\nPlease provide a detailed and comprehensive answer to the above question in the specified language, including specific visa or labor law information when relevant. Organize your response clearly and include all pertinent details.")
    ])
    
    chain = prompt | chat | StrOutputParser()
    response = await chain.ainvoke({
        "question": question,
        "language": language,
        "query_type": query_type,
        "contexts": contexts,
        "chat_history": formatted_chat_history
    })
    
    return response


# 14. 라우터 함수 정의
async def handle_visa_query(question: str, language: str, session_id: str, userNo: int, db: Session):
    return await handle_query(question, language, session_id, userNo, db, "Visa and Immigration")

async def handle_labor_law_query(question: str, language: str, session_id: str, userNo: int, db: Session):
    return await handle_query(question, language, session_id, userNo, db, "Labor Law")

async def handle_general_query(question: str, language: str, session_id: str, userNo: int, db: Session):
    return await handle_query(question, language, session_id, userNo, db, "General Information")

# Semantic Router 초기화
semantic_router = SemanticRouter({
    "Questions about visas, immigration, and stay in Korea": handle_visa_query,
    "Questions about labor laws, worker rights, and employment in Korea": handle_labor_law_query,
    "General questions about life in Korea": handle_general_query,
})

# 15. 입력 및 출력 모델 정의
# 세션 ID와 chat_his_no 매핑을 위한 딕셔너리
session_chat_mapping: Dict[str, int] = {}

# ChatRequest 모델 수정
class ChatRequest(BaseModel):
    question: str
    userNo: int
    categoryNo: int
    session_id: Optional[str] = None
    chat_his_no: Optional[int] = None
    is_new_session: Optional[bool] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    chatHisNo: int
    chatHisSeq: int
    detected_language: str



# 16. 채팅 엔드포인트
async def process_law_request(chat_request: ChatRequest, db: Session):
    try:
        question = chat_request.question
        userNo = chat_request.userNo
        categoryNo = chat_request.categoryNo
        session_id = chat_request.session_id or str(uuid.uuid4())
        chat_his_no = chat_request.chat_his_no

        language = detect_language(question)
        
        handler = semantic_router.route(question)
        
        response = await handler(question, language, session_id, userNo, db)
        
        is_new_session = chat_his_no is None and session_id not in session_chat_mapping
        current_chat_his_no = chat_his_no or session_chat_mapping.get(session_id)

        chat_history = crud.create_chat_history(
            db, 
            userNo, 
            categoryNo, 
            question, 
            response, 
            is_new_session=is_new_session, 
            chat_his_no=current_chat_his_no
        )
        
        session_chat_mapping[session_id] = chat_history.CHAT_HIS_NO

        async def generate_response():
            paragraphs = response.split('\n\n')
            for paragraph in paragraphs:
                words = paragraph.split()
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'content', 'text': word})}\n\n"
                    if i < len(words) - 1:
                        yield f"data: {json.dumps({'type': 'content', 'text': ' '})}\n\n"
                    await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'type': 'newline'})}\n\n"
                await asyncio.sleep(0.2)
            
            yield f"data: {json.dumps({'type': 'end', 'chatHisNo': chat_history.CHAT_HIS_NO, 'chatHisSeq': chat_history.CHAT_HIS_SEQ, 'detected_language': language})}\n\n"

        return StreamingResponse(generate_response(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )

    
# 세션 관리를 위한 새로운 엔드포인트 (차후 추가)
@app.post("/end_session")
async def end_session(session_id: str):
    if session_id in session_chat_mapping:
        del session_chat_mapping[session_id]
        if session_id in memory_store:
            del memory_store[session_id]
        return {"message": f"Session {session_id} has been ended and memory cleared."}
    return {"message": f"No active session found for {session_id}."}

# 17. 메모리 관리를 위한 새로운 엔드포인트 (차후 추가)
@app.post("/clear_memory")
async def clear_memory(session_id: str):
    if session_id in memory_store:
        del memory_store[session_id]
        return {"message": f"Memory for session {session_id} has been cleared."}
    return {"message": f"No memory found for session {session_id}."}

# 18. 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)