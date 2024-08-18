import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain.agents import Tool, AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from jobploy_crawler import jobploy_crawler
from pydantic import BaseModel
from typing import Union
from langchain.tools import StructuredTool
import re

# Load environment variables
load_dotenv()

class Query(BaseModel):
    query: str

# Initialize Elasticsearch client
elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
es_client = Elasticsearch([elasticsearch_url])

# Initialize OpenAI embeddings and chat model
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

vector_store = ElasticsearchStore(
    index_name="jobploy_jobs",
    embedding=embeddings,
    es_url=elasticsearch_url,
    distance_strategy="COSINE"  # 또는 "EUCLIDEAN" 또는 "DOT_PRODUCT"
)

def parse_pay(pay_str):
    # 시급, 월급, 연봉 구분
    pay_match = re.match(r"(\D+)\s*([\d,]+)\s*원", pay_str)
    if not pay_match:
        return None, None
    pay_type, pay_amount_str = pay_match.groups()
    pay_amount = int(pay_amount_str.replace(",", ""))

    # 단위에 따라 월급, 시급, 연봉 등을 구분
    if "시급" in pay_type:
        pay_type = "hourly"
    elif "월급" in pay_type:
        pay_type = "monthly"
    elif "연봉" in pay_type:
        pay_type = "annual"
    else:
        pay_type = "other"

    return pay_type, pay_amount

def index_job_data():
    job_data = jobploy_crawler('ko', pages=1)

    def process_batch(batch):
        texts = [f"{job['title']} - {job['location']} - {job['pay']} - {job['task']}" for job in batch]
        vectors = [embeddings.embed_query(text) for text in texts]
        metadatas = []

        for job in batch:
            pay_type, pay_amount = parse_pay(job['pay'])
            metadata = {
                'title': job['title'],
                'location': job['location'],
                'pay': job['pay'],
                'pay_type': pay_type,
                'pay_amount': pay_amount,
                'task': job['task'],
                'link': job['link']
            }
            metadatas.append(metadata)
        
        for vector, metadata in zip(vectors, metadatas):
            doc = {
                'text': metadata['title'],
                'vector': vector,
                'metadata': metadata
            }
            # print('========================================');
            # print(f"Indexing document: {doc}");
            # print('========================================');
            es_client.index(index='jobploy_jobs', body=doc)

    process_batch(job_data)

# Ensure the index exists and data is indexed
if es_client.indices.exists(index="jobploy_jobs"):
    es_client.indices.delete(index="jobploy_jobs")
index_job_data()

# Define search function
def search_jobs(query: Union[str, Query, dict]) -> str:
    if isinstance(query, Query):
        query_str = query.query
    elif isinstance(query, str):
        query_str = query
    elif isinstance(query, dict):
        query_str = query.get('query', '')
    else:
        raise ValueError(f"Invalid query type: {type(query)}. Expected str, Query object, or dict.")
    
    results = vector_store.similarity_search_with_score(query_str, k=5)
    return "\n".join([f"Title: {r[0].metadata['title']}\nLocation: {r[0].metadata['location']}\nPay: {r[0].metadata['pay']}\nTask: {r[0].metadata['task']}\nLink: {r[0].metadata['link']}\nScore: {r[1]}\n" for r in results])

# Create a tool for the agent to use
tools = [
    StructuredTool.from_function(
        func=search_jobs,
        name="JobSearch",
        description="Useful for searching job listings based on various criteria like location, salary, or job type.",
        args_schema=Query
    )
]

# Create the agent
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a specialized AI assistant for job searching and entity extraction. Your primary tasks are:


1. Help users find relevant job listings based on their queries using the JobSearch tool.

2. Extract and analyze key information from job listings and user queries, focusing on these entities:
   - **LOCATION**: Identify geographical locations (e.g., cities, provinces). Be precise in distinguishing between different administrative levels in South Korea:
     - "시" (si), "군" (gun), "구" (gu), "도" (do), and "읍" (eup) are common suffixes for cities, counties, districts, provinces, and towns, respectively.
     - Ensure to capture both the specific city and broader region (e.g., "서울특별시 강남구").
     - For locations written in English, ensure proper translation to or from Korean if necessary.
   - **MONEY**: Recognize salary or wage information and categorize it appropriately:
     - "시급" (hourly wage): Typically shown as 원/시간.
     - "월급" (monthly salary): Typically shown as 원/월.
     - "연봉" (annual salary): Typically shown as 원/년.
     - Clearly specify the type of compensation (hourly, monthly, or annual) when providing job details.
   - **OCCUPATION**: Detect job titles or professions from the "title" field of the job data. Provide a clear and concise summary of the job title.
   - **TASK**: Identify and summarize the main tasks or duties associated with the job position from the "task" field.
   - **LINK**: Include a link to the full job description for more details from the "link" field.
   - **CLOSING_DATE**: Recognize and inform the user about the closing date of the job listing, if available, which indicates how long the listing will remain open.

When responding to user queries:
1. First, use the JobSearch tool to find relevant job listings.
2. Then, analyze the results and the user's query to extract the key entities (LOCATION, MONEY, OCCUPATION, TASK, LINK, CLOSING_DATE).
3. Provide a summary of the job listings, highlighting the extracted entities:
   - Mention the specific type of salary or wage (시급, 월급, 연봉) and ensure clarity on the amount and period.
   - Clearly distinguish between cities, districts, and provinces to avoid confusion in the LOCATION information.
4. If the user's query lacks specificity, ask follow-up questions to refine the search based on location, salary expectations, specific job titles, or task-related keywords.

Always strive to provide accurate, relevant, and helpful information to assist users in their job search process."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Main interaction loop
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        response = agent_executor.run(user_input)
        print(f"Assistant: {response}")
