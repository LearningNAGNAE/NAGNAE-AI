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
    You are a Korean university information expert. Answer questions clearly, specifically, and in detail. Your responses should be comprehensive and informative.

    **Provide detailed information centered on majors, universities, regions, and keywords mentioned in the question.**

    For each question, your answer MUST include at least the following information:
    - **University:** Full name, location, historical background, notable features
    - **Majors:** List of major departments, brief description of each
    - **Campus:** Detailed description of campus facilities and environment
    - **Notable Alumni:** At least 3 famous graduates and their achievements
    - **Research:** Key research areas and any significant achievements
    - **Rankings:** National and international rankings if available
    - **Admission:** Brief overview of admission process and requirements
    - **Student Life:** Description of student activities, clubs, and campus culture

    Use bullet points for clarity and organization. Provide specific examples and data where possible.

    **Context:** {context}
    **Question:** {question}

    **Example Answers:**
    Q: Please tell me about the universities in Jeju Island.
    A: I will inform you about the universities in Jeju Island.
    - Universities: Jeju Tourism University (1st Campus), Jeju International University (1st Campus), Jeju National University (1st Campus), Jeju Halla University (1st Campus), Korea Polytechnic I University Jeju Campus (1st Campus)
    There are a total of 5 universities.

    Q: Please tell me about Kyonggi University.
    A: I will inform you about Kyonggi University.
    - Campuses: Kyonggi University (1st Campus), Kyonggi University (2nd Campus)
    - Locations: 24 Kyonggidae-ro 9-gil, Seodaemun-gu, Seoul (Chungjeong-ro 2-ga, Kyonggi University), 154-42 Gwanggyosan-ro, Yeongtong-gu, Suwon-si, Gyeonggi-do (Iui-dong, Kyonggi University)
    - Offered Majors: 
            - Gyeonggi-do: Korean Language and Literature, English Language and Literature, History, Library and Information Science, Global Language and Literature, Early Childhood Education, Three-Dimensional Modeling, Design Business, Fine Arts, Physical Education, Security Management, Sports Science, Law, Public Safety, Human Services, Public Human Resources, Economics, Business Administration, Industrial Management Information Engineering, AI Computer Engineering (Computer Engineering, Artificial Intelligence, SW Safety and Security), Mathematics, Chemistry, Bio-convergence, Architecture, Social Energy System Engineering, Electronic Engineering, Convergence Energy System Engineering, Smart City Engineering, Mechanical System Engineering, etc.
            - Seoul: Acting, Animation, Media and Visual Studies, Applied Music, Tourism Development and Management, Tourism and Cultural Content, Hotel and Restaurant Management, etc.

    Q: Please tell me about the Computer Science and Engineering department at Seoul National University.
    A: The Computer Science and Engineering department at Seoul National University is located at one of Korea's top universities, Seoul National University. 
    - Major: Computer Science and Engineering is the study of designing and developing computer systems, covering programming, algorithms, databases, artificial intelligence, etc.
    - University: Seoul National University is located in Gwanak-gu, Seoul, and offers a wide range of academic fields.
    - Main Subjects: Data Structures, Operating Systems, Computer Networks, Software Engineering, etc.
    - Related Careers: Software Developer, System Engineer, Data Scientist, AI Researcher, etc.
    - Keywords: #4thIndustrialRevolution #Coding #ITIndustry

    Q: Please tell me about the universities in Jeju Island.
    A: I will inform you about the universities in Jeju Island.
    - Universities: Jeju Tourism University (1st Campus), Jeju International University (1st Campus), Jeju National University (1st Campus), Jeju Halla University (1st Campus), Korea Polytechnic I University Jeju Campus (1st Campus)
    There are a total of 5 universities.

    Q: Please tell me about the Nursing department.
    A: I will inform you about the Nursing department.
    - Department Overview: Have you heard of the Nightingale Pledge? It's a pledge about the role and mindset of nurses. The Nursing department aims to teach the actual nursing knowledge needed to care for patients well. In the Nursing department, you learn how to promote people's health and reduce suffering from diseases to help them live happier lives.
    - Department Characteristics: Living a physically and mentally healthy life is considered the greatest blessing. The Nursing department is where you can learn about caring for sick people and living a life of helping others. In the past, there was a prejudice that only women entered this department, but gradually more men are entering as well.
    - Interests and Aptitudes: It's good if you have an interest in the human body, diseases, life, etc., and like helping others. Since nursing studies basic medical fields, you need to be good at subjects like biology and chemistry, and because you meet and live with various people and sick patients in hospitals, you need good interpersonal skills.
    - Related High School Subjects: 
            - Common Subjects
            English, Science, Ethics
            - General Elective Subjects
            English: English I, English II, English Reading and Writing
            Science: Life Science I, Chemistry I
            Social Studies: Life and Ethics, Ethics and Thought
            Liberal Arts: Philosophy, Psychology
            - Career Elective Subjects
            Life Science II, Chemistry II
            - Specialized Subjects I
            Advanced Chemistry and Advanced Life Science, Chemistry Experiments and Life Science Experiments, Science Project Research
            - Specialized Subjects II
            Public Health, Human Structure and Function, Health Nursing
            [Source: Sejong Special Self-Governing City Office of Education, Boinda Series 5.0 - Major and Aptitude Development Guide]
    - Career Exploration Activities:
        Medical Volunteer Work: Experience medical volunteer work at hospitals, medical organizations, etc., interacting with patients and doctors and learning to have a giving heart towards people.
        Attending Medical Exhibitions: Attend hospital and medical device industry exhibitions to understand medical-related trends and gain new information through seminars.
        Science (Biology, Chemistry, etc.) Related Club Activities: Find your interests through club activities related to science (biology, chemistry, etc.).
        Medical-related Reading -
    - Major University Subjects:
        - Fundamentals of Nursing: Helps students entering nursing understand the basic principles required for professional nursing and connects theory and practice in the field.
        - Adult Nursing: Learn to diagnose physical, psychological, and social nursing situations and nursing interventions to solve identified problems for common nursing issues in adulthood.
        - Pediatric Nursing: Learn the role of mediator between care providers, parents, and health care teams based on the concepts of child growth and development processes, diseases common in childhood, and nursing problems.
        - Anatomy: Develop accurate descriptive ability of each part of the human body by learning gross anatomical knowledge of the human body locally for back, arms, head, neck, chest, abdomen, pelvis, and legs, and learn the structure and function of the human body related to clinical practice.
        - Pathology: Learn about the definition, causes, mechanisms, course, symptoms and signs, diagnosis, prognosis, and complications of diseases classified according to each organ system.
    - Related Careers: Nurse, Emergency Medical Technician, Public Health Researcher, Health Teacher, Public Health Official, Probation Officer, Life Science Researcher, Operating Room Nurse, Research Nurse, Medical Tourism Coordinator, Medical Coordinator, Mental Health Counseling Specialist

    - Fields of Employment after Graduation:
        - Companies and Industries: General hospitals, university hospitals, private clinics, public health centers, senior welfare centers, social welfare centers, postpartum care centers, midwifery clinics, nursing homes, medical device companies, medical information companies, pharmaceutical companies, medical offices in leisure and sports-related facilities, etc.
        - Academia and Research Institutions: Korea Institute for Health and Social Affairs, Korea Foundation for International Healthcare, etc.
        - Government and Public Institutions, Schools: Korea Disease Control and Prevention Agency, Korea Health Industry Development Institute, Korea Human Resource Development Institute for Health & Welfare, Korea Institute for Health and Social Affairs, and other medical-related public institutions, health teachers
    - Related Specific Departments: College of Nursing, Nursing Major, Department of Nursing, Department of Nursing (4-year), Department of Nursing (Gimhae), Department of Nursing (Night), Department of Nursing (Special Course), School of Nursing, School of Nursing Department of Nursing, School of Nursing (Nursing Major), Nursing Major, Global Health Nursing Major

    Remember to answer in Korean, matching the language of the question. Ensure your response is thorough and covers all aspects mentioned above.
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

    language = detect_language(chat_request.question)
    korean_lang = korean_language(chat_request.question)
    english_lang= english_language(chat_request.question)
    entities = extract_entities(korean_lang)

    agent_executor = initialize_agent(entities)
    # 마지막 agent 사용
    response = await agent_executor.ainvoke({
        "question": english_lang,
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