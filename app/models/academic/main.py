from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cross_encoder, openai, es_client, embedding, session_histories, fine_tuned_model, fine_tuned_tokenizer
from embedding_indexing import index_exists, embed_and_index_university_data, embed_and_index_university_major, embed_and_index_major_details, embed_and_index_pdf_data, update_indices
from utils import detect_language, korean_language, extract_entities, generate_elasticsearch_query, generate_response_with_fine_tuned_model
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List, Optional
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    input: str
    session_id: str

class Response(BaseModel):
    answer: str

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
    You are a Korean university information expert. Your role is to provide accurate and detailed answers to questions about Korean universities using the provided tools.

    Information Provision:
    - Answer questions regarding university admission procedures, programs, majors, and related information.
    - Focus your responses using the extracted entities (universities, majors, keywords).

    Language and Translation:
    - Translate the final response into {language}. Always ensure that the response is translated, and if it is not, make sure to translate it again.
    - Provide only the translated response.

    Structure and Clarity:
    - Present your answers clearly and in an organized manner. Use bullet points or numbered lists if necessary.
    - Include examples or scenarios to illustrate how the information applies.

    Accuracy and Updates:
    - Provide accurate information based on the latest data available from the tools.
    - Advise the user to check official sources for the most current information.

    Extracted Entities:
    - Universities: {universities}
    - Majors: {majors}
    - Keywords: {keywords}

    Use these entities to guide your search and response.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "language", "universities", "majors", "keywords"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x["question"],
            "language": lambda x: x["language"],
            "universities": lambda x: ", ".join(x["universities"]),
            "majors": lambda x: ", ".join(x["majors"]),
            "keywords": lambda x: ", ".join(x["keywords"]),
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | (lambda x: generate_response_with_fine_tuned_model(str(x), fine_tuned_model, fine_tuned_tokenizer))
        | StrOutputParser()
    )

    return qa_chain

@app.post("/academic", response_model=Response)
async def query_agent(query: Query):
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
    language = detect_language(query.input)
    korean_lang = korean_language(query.input)
    entities = extract_entities(korean_lang)

    # 세션 기록 가져오기 또는 새로 생성
    chat_history = session_histories.get(query.session_id, [])

    response = await agent_executor.ainvoke({
        "question": query.input,
        "chat_history": chat_history,
        "agent_scratchpad": [],
        "language": language,
        "universities": entities.universities,
        "majors": entities.majors,
        "keywords": entities.keywords
    })

    # 대화 기록 업데이트
    chat_history.append({"role": "user", "content": query.input})
    chat_history.append({"role": "assistant", "content": response})
    session_histories[query.session_id] = chat_history

    return Response(answer=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)