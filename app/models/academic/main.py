from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cross_encoder, openai, es_client, embedding 
from embedding_indexing import index_exists, embed_and_index_university_data, embed_and_index_university_major, embed_and_index_major_details, embed_and_index_pdf_data, update_indices
from utils import trans_language, detect_language, korean_language, extract_entities, generate_elasticsearch_query, english_language
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import uvicorn
import uuid
from sqlalchemy.orm import Session
# 1. 환경 설정
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(app_dir)
from ...database.db import get_db
from ...database import crud



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 ID와 chat_his_no 매핑을 위한 딕셔너리
session_chat_mapping: Dict[str, int] = {}

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

async def multi_index_search(query, entities, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=100):
    """멀티 인덱스 검색 함수"""
    if isinstance(query, dict):
        query = query.get('question', '')
    # entities = extract_entities(query)
    query_vector = embedding.embed_query(query)
    es_query = generate_elasticsearch_query(entities)
    
    index_weights = {
        'university_data': 0.35,
        'university_major': 0.3,
        'major_details': 0.25,
        'pdf_data': 0.1
    }

    multi_search_body = []
    for index in indices:
        search_body = {
            "size": top_k * 2,
            "query": {
                "function_score": {
                    "query": es_query["query"],
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "source": f"cosineSimilarity(params.query_vector, 'vector') * {index_weights.get(index, 1.0)} + 1.0",
                                    "params": {"query_vector": query_vector}
                                }
                            }
                        }
                    ],
                    "boost_mode": "multiply"
                }
            },
            "_source": ["text", "metadata"]
        }
        multi_search_body.extend([{"index": index}, search_body])

    results = es_client.msearch(body=multi_search_body)

    processed_results = []
    for i, response in enumerate(results['responses']):
        if response['hits']['hits']:
            for hit in response['hits']['hits']:
                processed_results.append({
                    'index': indices[i],
                    'score': hit['_score'],
                    'text': hit['_source']['text'],
                    'metadata': hit['_source']['metadata']
                })

    # Reranking using CrossEncoder
    if processed_results:
        rerank_input = [(query, result['text']) for result in processed_results]
        rerank_scores = cross_encoder.predict(rerank_input)
        
        for i, score in enumerate(rerank_scores):
            processed_results[i]['rerank_score'] = score

        processed_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    print(processed_results[:top_k]);
    return processed_results[:top_k]

def initialize_agent(entities):
    """에이전트 초기화 함수"""
    class FunctionRetriever(BaseRetriever):
        async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
            if isinstance(query, dict):
                query = query.get('question', '')
            results = await multi_index_search(query, entities, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=4)
            return [Document(page_content=result['text'], metadata={**result['metadata'], 'rerank_score': result.get('rerank_score', 0)}) for result in results]

        async def get_relevant_documents(self, query: str) -> List[Document]:
            return await self._aget_relevant_documents(query)
        
        
    retriever = FunctionRetriever()

    prompt_template = """
    You are a Korean university information expert. Answer questions clearly, specifically, and in detail in Korean. Your responses should be comprehensive and informative.

    **Provide detailed information centered on majors, universities, regions, and keywords mentioned in the question.**

    Use the following information sources to answer the question:
    1. University Data: {university_data}
    2. University Major Data: {university_major}
    3. Major Details: {major_details}
    4. Foreign Student Information: {pdf_data}

    IMPORTANT: 
    1. Do not invent or generate any information that is not present in the provided data. If information is not available, explicitly state "이 정보는 제공된 데이터에 없습니다." Also, provide a relevant university website link if available. Example: "이 정보는 제공된 데이터에 없습니다. 자세한 내용은 다음 웹사이트를 참조하세요: [대학 웹사이트 링크]."
    2. Pay close attention to the exact names of universities and locations. Do not confuse similar-sounding names (e.g., Cheongju University vs Chungju University). 
    3. If a location name is mentioned in the question, do not assume it is a university name. Always check if it's a university in the provided data.
    4. If you're unsure about any name or information, state clearly: "정확한 정보를 확인할 수 없습니다. 제공된 데이터에서 [name/information]에 대한 정보를 찾을 수 없습니다."
    
    For each question, your answer MUST include the following information (provide in an appropriate order based on the nature of the question):
    - **University:** Full name, location, historical background, notable features (only provide this information if explicitly asked about university details. If asked about specific majors, recommend universities known for those majors only if relevant information is available; otherwise, do not include it. For all other situations, do not provide university information unless explicitly requested)
    - **Majors:** Provide information only about the specific major(s) the user asks about, and give a brief description of each. Additionally, provide information about universities known for those specific majors
    - **Campus:** Detailed description of campus facilities and environment. List the number of campuses and their respective names for the university
    - **Research:** Key research areas and any significant achievements
    - **Admission:** Brief overview of admission process and requirements (provide separate explanations for the admission process and requirements for international students and general applicants)
    - **Student Life:** Description of student activities, clubs, and campus culture

    Use bullet points for clarity and organization. Provide specific examples and data where possible.

    Utilize the foreign student information ({pdf_data}) when providing details about special programs for international students, language requirements, cultural adaptation support, etc.

    **Question:** {question}

    Always answer in Korean. Ensure your response is thorough and covers all aspects mentioned above. You can include English terms or names in parentheses if necessary. Do not include any information that is not present in the provided data sources. If the question mentions a location or name that is not clearly a university in the provided data, clarify this in your response and provide information about the location if available, not about a university.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["university_data", "university_major", "major_details", "pdf_data", "question"]
    )

    async def format_docs(docs):
        formatted = {
            "university_data": "",
            "university_major": "",
            "major_details": "",
            "pdf_data": ""
        }
        for doc in docs:
            source = doc.metadata.get('source_index', 'unknown')
            if source in formatted:
                formatted[source] += f"\n{doc.page_content}"
        return formatted

    async def get_formatted_docs(question, index):
        docs = await retriever.get_relevant_documents(question)
        formatted = await format_docs(docs)
        return formatted[index]

    qa_chain = (
        {
            "university_data": lambda x: get_formatted_docs(x["question"], "university_data"),
            "university_major": lambda x: get_formatted_docs(x["question"], "university_major"),
            "major_details": lambda x: get_formatted_docs(x["question"], "major_details"),
            "pdf_data": lambda x: get_formatted_docs(x["question"], "pdf_data"),
            "question": lambda x: x["question"]
        }
        | prompt
        | (lambda x: openai.predict(str(x)))  # GPT-3.5-turbo 모델로 예측
        | StrOutputParser()
    )

    return qa_chain










@app.post("/academic", response_model=ChatResponse)
async def query_agent(request: Request, chat_request: ChatRequest, db: Session = Depends(get_db)):

    question = chat_request.question
    userNo = chat_request.userNo
    categoryNo = chat_request.categoryNo
    session_id = chat_request.session_id or str(uuid.uuid4())
    chat_his_no = chat_request.chat_his_no
    is_new_session = chat_request.is_new_session

    # 인덱스 초기화 확인 및 수행
    if not index_exists('university_data') or \
       not index_exists('university_major') or \
       not index_exists('major_details') or \
       not index_exists('pdf_data'):
        print("Initial setup required. Running full indexing process...")
        await embed_and_index_university_data()
        await embed_and_index_university_major()
        await embed_and_index_major_details()
        await embed_and_index_pdf_data()
        print("Initial indexing completed.")

    # await update_indices()

    language = detect_language(chat_request.question)
    korean_lang = korean_language(chat_request.question)
    english_lang= english_language(chat_request.question)
    entities = extract_entities(korean_lang)

    agent_executor = initialize_agent(entities)
    # 마지막 agent 사용
    response = await agent_executor.ainvoke({
        "question": korean_lang,
        "agent_scratchpad": [],
        "universities": entities.universities,
        "majors": entities.majors,
        "regions": entities.regions,
        "keywords": entities.keywords
    })

    # 번역기
    translated_response = trans_language(response, language)

    # 채팅 기록 저장
    chat_history = crud.create_chat_history(db, userNo, categoryNo, question, translated_response, is_new_session, chat_his_no)
    
    # 세션 ID와 chat_his_no 매핑 업데이트
    session_chat_mapping[session_id] = chat_history.CHAT_HIS_NO



    chat_response = ChatResponse(
            question=question,
            answer=translated_response,
            chatHisNo=chat_history.CHAT_HIS_NO,
            chatHisSeq=chat_history.CHAT_HIS_SEQ,
            detected_language=language
        )

    return JSONResponse(content=chat_response.dict())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)