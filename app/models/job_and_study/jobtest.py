# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.agents import create_openai_tools_agent, AgentExecutor
# from langchain.tools.retriever import create_retriever_tool
# from langchain import hub
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory
# import os
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import spacy

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define the Pydantic model for request validation
# class QueryRequest(BaseModel):
#     query: str

# # Initialize Spacy model for NER
# nlp = spacy.load("ko_core_news_sm")

# def extract_entities(query):
#     doc = nlp(query)
#     entities = {"LOCATION": [], "MONEY": [], "OCCUPATION": []}
    
#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)
    
#     # Keywords and major areas (could be moved to a config file or database)
#     location_keywords = {
#         "서울": "서울특별시", "경기": "경기도", "인천": "인천광역시", "부산": "부산광역시", 
#         "대구": "대구광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", 
#         "세종": "세종특별자치시", "강원": "강원도", "충북": "충청북도", "충남": "충청남도", 
#         "전북": "전라북도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", 
#         "제주": "제주특별자치도"
#     }
#     major_areas = {
#         "수원": "시", "성남": "시", "안양": "시", "안산": "시", "용인": "시", "부천": "시", 
#         "광명": "시", "평택": "시", "과천": "시", "오산": "시", "시흥": "시", "군포": "시", 
#         "의왕": "시", "하남": "시", "이천": "시", "안성": "시", "김포": "시", "화성": "시", 
#         "광주": "시", "여주": "시", "고양": "시", "의정부": "시", "파주": "시", "양주": "시", 
#         "구리": "시", "남양주": "시", "포천": "시", "동두천": "시",
#         "가평": "군", "양평": "군", "연천": "군",
#         "청주": "시", "충주": "시", "제천": "시", "보은": "군", "옥천": "군", "영동": "군", 
#         "증평": "군", "진천": "군", "괴산": "군", "음성": "군", "단양": "군",
#         "천안": "시", "공주": "시", "보령": "시", "아산": "시", "서산": "시", "논산": "시", 
#         "계룡": "시", "당진": "시", "금산": "군", "부여": "군", "서천": "군", "청양": "군", 
#         "홍성": "군", "예산": "군", "태안": "군",
#         "전주": "시", "군산": "시", "익산": "시", "정읍": "시", "남원": "시", "김제": "시", 
#         "완주": "군", "진안": "군", "무주": "군", "장수": "군", "임실": "군", "순창": "군", 
#         "고창": "군", "부안": "군",
#         "목포": "시", "여수": "시", "순천": "시", "나주": "시", "광양": "시", "담양": "군", 
#         "곡성": "군", "구례": "군", "고흥": "군", "보성": "군", "화순": "군", "장흥": "군", 
#         "강진": "군", "해남": "군", "영암": "군", "무안": "군", "함평": "군", "영광": "군", 
#         "장성": "군", "완도": "군", "진도": "군", "신안": "군",
#         "포항": "시", "경주": "시", "김천": "시", "안동": "시", "구미": "시", "영주": "시", 
#         "영천": "시", "상주": "시", "문경": "시", "경산": "시", "군위": "군", "의성": "군", 
#         "청송": "군", "영양": "군", "영덕": "군", "청도": "군", "고령": "군", "성주": "군", 
#         "칠곡": "군", "예천": "군", "봉화": "군", "울진": "군", "울릉": "군",
#         "창원": "시", "진주": "시", "통영": "시", "사천": "시", "김해": "시", "밀양": "시", 
#         "거제": "시", "양산": "시", "의령": "군", "함안": "군", "창녕": "군", "고성": "군", 
#         "남해": "군", "하동": "군", "산청": "군", "함양": "군", "거창": "군", "합천": "군",
#         "제주": "시", "서귀포": "시"
#     }
    
#     words = query.split()
#     for word in words:
#         if word in location_keywords:
#             entities["LOCATION"].append(location_keywords[word])
#         elif word.endswith(("시", "군", "구")) or word in major_areas:
#             entities["LOCATION"].append(word + major_areas.get(word, ""))
    
#     return entities

# def configure_chrome_driver():
#     chrome_driver_path = r"C:\chromedriver\chromedriver.exe"
#     chrome_options = Options()
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_argument("--window-size=1920x1080")
#     return webdriver.Chrome(service=Service(chrome_driver_path), options=chrome_options)

# def jobploy_crawler(pages=5):
#     driver = configure_chrome_driver()
#     results = []
    
#     try:
#         for page in range(1, pages + 1):
#             driver.get(f"https://www.jobploy.kr/ko/recruit?page={page}")
#             WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "content")))

#             job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

#             for job in job_listings:
#                 title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
#                 link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
#                 salary_element = job.find_element(By.CSS_SELECTOR, ".pay")
                
#                 badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")
#                 location = badge_elements[0].text if len(badge_elements) > 0 else "위치 정보 없음"
#                 task = badge_elements[1].text if len(badge_elements) > 1 else "직무 정보 없음"
#                 closing_date = badge_elements[2].text if len(badge_elements) > 2 else "마감 정보 없음"
                
#                 title = title_element.text
#                 salary = salary_element.text

#                 results.append({
#                     "title": title,
#                     "link": link_element,
#                     "location": location,
#                     "task": task,
#                     "salary": salary,
#                     "closing_date": closing_date
#                 })
#     finally:
#         driver.quit()
    
#     return results

# def prepare_documents():
#     results = jobploy_crawler()
#     documents = []

#     for idx, result in enumerate(results):
#         unique_id = result['link'] if 'link' in result and result['link'] else str(idx)
#         full_text = (f"ID: {unique_id}\nTitle: {result['title']}\nLocation: {result['location']}\n"
#                      f"Task: {result['task']}\nSalary: {result['salary']}\nClosing Date: {result['closing_date']}\n"
#                      f"Link: {result['link']}")
#         metadata = {
#             "link": result['link'],
#             "location": result['location'],
#             "task": result['task'],
#             "salary": result['salary'],
#             "closing_date": result['closing_date'],
#             "unique_id": unique_id
#         }
#         documents.append({"text": full_text, "metadata": metadata})

#     faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings, metadatas=[doc["metadata"] for doc in documents])
#     return faiss_index

# faiss_index = prepare_documents()
# retriever = faiss_index.as_retriever(search_type="similarity")
# retriever_tool = create_retriever_tool(
#     retriever, 
#     "job_search", 
#     "Use this tool to search for job information from Jobploy website."
# )

# memory = ConversationBufferMemory(memory_key="chat_history")

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", """
#             You are a specialized AI assistant focused on job searches in Korea. Your primary function is to provide accurate, relevant, and up-to-date job information based on user queries.

#             You utilize the following NER categories to extract key information from user queries:
#             - LOCATION: Identifies geographical locations (e.g., cities, provinces)
#             - MONEY: Recognizes salary or wage information
#             - OCCUPATION: Detects job titles or professions

#             When responding to queries:
#             1. Analyze user queries to extract relevant entities.
#             2. Search the job database using the extracted information.
#             3. Filter and prioritize job listings based on the user's requirements.
#             4. Provide a comprehensive summary of the search results.
#             5. Offer detailed information about each relevant job listing.

#             Include the following information for each job listing:
#             - Title
#             - Company (if available)
#             - Location
#             - Task
#             - Salary information
#             - Brief job description (if available)
#             - Key requirements (if available)
#             - Application link

#             Ensure your response is clear, concise, and directly addresses the user's query.
#         """),
#         ("user", "I need you to search for job listings based on my query. Can you help me with that?"),
#         ("assistant", "Certainly! I'd be happy to help you search for job listings based on your query. Please provide me with your specific request, and I'll use the search_jobs tool to find relevant information for you. What kind of job or criteria are you looking for?"),
#         ("user", "{input}"),
#         ("assistant", "Thank you for your query. I'll search for job listings based on your request using the search_jobs tool. I'll provide a summary of the results and detailed information about relevant job listings."),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# agent = create_openai_tools_agent(llm=chat_openai, tools=[retriever_tool], prompt=prompt, memory=memory)
# agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

# @app.get("/")
# async def read_root():
#     return {"message": "Welcome to the JobPloy API"}

# @app.post("/ask")
# async def ask(query_request: QueryRequest):
#     user_query = query_request.query

#     if not user_query:
#         raise HTTPException(status_code=400, detail="Query is required")

#     entities = extract_entities(user_query)

#     try:
#         result = agent_executor.invoke({"input": user_query})

#         response_with_links = {"answer": "Here are the jobs that match your query:<br>"}
#         source_documents = result.get("source_documents", [])
#         for doc in source_documents:
#             title = doc.page_content
#             link = doc.metadata.get("link", "https://www.jobploy.kr/ko/recruit")
#             location = doc.metadata.get("location", "위치 정보 없음")
#             task = doc.metadata.get("task", "직무 정보 없음")
#             salary = doc.metadata.get("salary", "급여 정보 없음")
#             closing_date = doc.metadata.get("closing_date", "마감 정보 없음")
#             response_with_links["answer"] += (f"<br>- {title} (위치: {location}, 직무: {task}, "
#                                               f"급여: {salary}, 마감일: {closing_date}): {link}")

#         return response_with_links

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)