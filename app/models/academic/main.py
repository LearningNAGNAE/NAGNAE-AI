from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cross_encoder, openai, es_client, embedding, session_histories, fine_tuned_model, fine_tuned_tokenizer
from embedding_indexing import index_exists, embed_and_index_university_data, embed_and_index_university_major, embed_and_index_major_details, embed_and_index_pdf_data, update_indices
from utils import trans_language, detect_language, korean_language, extract_entities, generate_elasticsearch_query, generate_response_with_fine_tuned_model
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

async def multi_index_search(query, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=10):
    """멀티 인덱스 검색 함수"""
    if isinstance(query, dict):
        query = query.get('question', '')
    
    entities = extract_entities(query)
    query_vector = embedding.embed_query(query)
    es_query = generate_elasticsearch_query(entities)
    
    index_weights = {
        'university_data': 0.4,
        'university_major': 0.3,
        'major_details': 0.2,
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

    return processed_results[:top_k]

def initialize_agent():
    """에이전트 초기화 함수"""
    class FunctionRetriever(BaseRetriever):
        async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
            if isinstance(query, dict):
                query = query.get('question', '')
            results = await multi_index_search(query, indices=['university_data', 'university_major', 'major_details', 'pdf_data'], top_k=10)
            return [Document(page_content=result['text'][:500], metadata={**result['metadata'], 'rerank_score': result.get('rerank_score', 0)}) for result in results]

        async def get_relevant_documents(self, query: str) -> List[Document]:
            return await self._aget_relevant_documents(query)
        
        
    retriever = FunctionRetriever()

    prompt_template = """
    당신은 한국 대학 정보 전문가입니다. 한국 대학에 대한 질문에 정확하고 상세한 답변을 제공하는 역할을 맡고 있습니다. 제공된 도구를 사용하여 대학 입학 절차, 프로그램, 전공 및 관련 정보에 대한 자세한 답변을 제공하십시오.

    **정보 제공:**
    - 대학 입학 절차, 학술 프로그램, 전공 및 관련 정보를 포괄적으로 제공하십시오.
    - 질문에 언급된 지역에 위치한 모든 대학의 정보를 생략하지 말고 포함하십시오.
    - 추출된 엔티티인 대학, 전공, 지역 및 키워드를 활용하여 답변을 집중하고 향상시키십시오.

    **구조와 명확성:**
    - 답변을 명확하고 체계적으로 제시하십시오.
    - 필요에 따라 총알 점이나 번호 목록을 사용하여 정보를 정리하십시오.
    - 정보의 적용 예시나 시나리오를 포함하십시오.

    **정확성 및 업데이트:**
    - 제공된 정보가 도구에서 제공하는 최신 데이터에 기반하여 정확한지 확인하십시오.
    - 공식 대학 웹사이트나 기타 권위 있는 출처를 통해 세부 정보를 확인하도록 권장하십시오.

    **추출된 엔티티:**
    - **대학:** 질문과 관련된 대학 이름을 나열하십시오. 해당 지역의 모든 대학을 포함하십시오.
    - **전공:** 언급된 경우 특정 학술 프로그램이나 전공에 대한 세부 정보를 제공하십시오.
    - **지역:** 질문과 관련된 지역에 대한 세부 정보를 제공하십시오. 지역은 대학교의 위치를 기준으로 하여 제공하되, 관련 지역의 설명도 포함하십시오.
    - **키워드:** 추가적인 맥락이나 세부 정보를 제공하기 위해 관련 키워드를 포함하십시오.

    **컨텍스트:** {context}

    **질문:** {question}

    **답변:**
    - 질문의 모든 측면을 다루는 자세한 답변을 제공하십시오.
    - 필요한 경우 이름, 위치, 캠퍼스 정보 및 공식 웹사이트 링크와 같은 특정 세부 정보를 포함하십시오.
    - 지역에 대한 답변은 대학교의 위치를 기준으로 하여, 관련 지역의 설명과 함께 제공하십시오.
    - 사용된 정보의 출처를 인용하여 신뢰성을 보장하십시오.

    **예시 질문:** '경기도에 있는 대학교들을 생략하지 말고 알려줘 자료 출처도 알려줘'

    **예시 답변:**
    - **고려대학교**
    - **위치:** 서울특별시 성북구 안암동
    - **캠퍼스 정보:** 안암캠퍼스, 세종캠퍼스
    - **웹사이트:** [고려대학교](http://www.korea.ac.kr)
    - **지역 설명:** 경기도와 인접한 서울특별시에 위치한 고려대학교는 한국을 대표하는 대학 중 하나로, 국내외에서 평가가 높은 대학입니다.

    - **한국외국어대학교**
    - **위치:** 경기도 용인시 수지구 죽전로 55
    - **캠퍼스 정보:** 제1캠퍼스, 제2캠퍼스
    - **웹사이트:** [한국외국어대학교](http://www.hufs.ac.kr)
    - **지역 설명:** 경기도 용인시에 위치한 한국외국어대학교는 외국어 교육 및 국제 교류에 특화된 대학으로, 다양한 언어 전공 프로그램을 제공합니다.

    - **경희대학교**
    - **위치:** 경기도 용인시 처인구 포곡읍 서울대학로 1732
    - **캠퍼스 정보:** 서울캠퍼스, 국제캠퍼스
    - **웹사이트:** [경희대학교](http://www.khu.ac.kr)
    - **지역 설명:** 경기도 용인시에 위치한 경희대학교는 국내 최고의 사립대학 중 하나로, 교육 및 연구 분야에서 우수한 성과를 내고 있습니다.

    **출처:** 최신 데이터에서 제공된 정보입니다. 가장 정확하고 최신의 세부 사항은 공식 대학 웹사이트를 참조하십시오.

    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "universities", "majors", "regions", "keywords"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x["question"],
            "universities": lambda x: ", ".join(x["universities"]),
            "majors": lambda x: ", ".join(x["majors"]),
            "regions": lambda x: ", ".join(x["regions"]),
            "keywords": lambda x: ", ".join(x["keywords"])
        }
        | prompt
        # | (lambda x: generate_response_with_fine_tuned_model(str(x), fine_tuned_model, fine_tuned_tokenizer))
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

    agent_executor = initialize_agent()
    language = detect_language(chat_request.question)
    korean_lang = korean_language(chat_request.question)
    entities = extract_entities(korean_lang)


    # 마지막 agent 사용
    response = await agent_executor.ainvoke({
        "question": korean_lang,
        "agent_scratchpad": [],
        "universities": entities.universities,
        "majors": entities.majors,
        "regions": entities.region,
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