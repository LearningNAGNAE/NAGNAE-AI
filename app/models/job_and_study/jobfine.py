from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_core.documents import Document
import json
import os
import re
from dotenv import load_dotenv
from langid import classify
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
from datasets import Dataset, load_metric
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
import numpy as np

# Load environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Load tokenizer and model for BERT
model_name = "bert-base-multilingual-cased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model config and set number of labels (assuming 4 labels for NER: LOC, MONEY, OCCUPATION, CLOSINGDATE)
num_labels = 4
config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels

# Load model with updated config
model = BertForTokenClassification.from_pretrained(model_name, config=config)

# NER pipeline setup
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Define label to index mapping
label_to_index = {
    "LOC": 0,
    "MONEY": 1,
    "OCCUPATION": 2,
    "CLOSINGDATE": 3
}

# Function to load and label data from crawling results
def load_and_label_crawled_data(lang):
    # Load the crawled data for the given language
    filename = f"crawled_data_{lang}.txt"
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    labeled_data = []
    for item in data:
        tokens = tokenizer.tokenize(item['title'] + " " + item['task'])
        labels = [-100] * len(tokens)

        # Label the data based on detected entities
        for entity in item.get('entities', []):
            entity_tokens = tokenizer.tokenize(entity['text'])
            for i, token in enumerate(tokens):
                if token in entity_tokens:
                    labels[i] = label_to_index[entity['label']]

        labeled_data.append({"tokens": tokens, "labels": labels})

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({
        "tokens": [item['tokens'] for item in labeled_data],
        "labels": [item['labels'] for item in labeled_data]
    })
    
    return dataset

# Tokenization and alignment function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

# Define metrics for evaluation
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_to_index[pred] for (pred, label) in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_to_index[label] for (pred, label) in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Function to fine-tune the model based on language
def fine_tune_model_for_language(lang):
    dataset = load_and_label_crawled_data(lang)
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # Fine-tuning setup
    training_args = TrainingArguments(
        output_dir=f"./results_{lang}",
        evaluation_strategy="epoch",
        learning_rate=4e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluation
    results = trainer.evaluate()
    print(f"Results for language {lang}:", results)

# Example: fine-tune model for Korean language
fine_tune_model_for_language("ko")

# Entity extraction function using the fine-tuned model
def extract_entities(query):
    ner_results = ner_pipeline(query)
    entities = {
        "LOCATION": [],
        "MONEY": [],
        "OCCUPATION": [],
        "CLOSINGDATE": []
    }

    current_entity = ""
    current_label = ""

    for entity in ner_results:
        entity_label = entity['entity_group']
        entity_text = entity['word']

        if entity_text.startswith("##"):
            current_entity += entity_text[2:]
        else:
            if current_entity:
                if current_label == "LOC":
                    entities["LOCATION"].append(current_entity)
                elif current_label == "MONEY":
                    entities["MONEY"].append(current_entity)
                elif current_label == "OCCUPATION":
                    entities["OCCUPATION"].append(current_entity)
                elif current_label == "CLOSINGDATE":
                    entities["CLOSINGDATE"].append(current_entity)
            current_entity = entity_text
            current_label = entity_label

    if current_entity:
        if current_label == "LOC":
            entities["LOCATION"].append(current_entity)
        elif current_label == "MONEY":
            entities["MONEY"].append(current_entity)
        elif current_label == "OCCUPATION":
            entities["OCCUPATION"].append(current_entity)
        elif current_label == "CLOSINGDATE":
            entities["CLOSINGDATE"].append(current_entity)

    return entities

# Sample query for testing
test_query = "경기도 화성시에서 300만원 이상의 월급을 받는 일을 찾고 있습니다. 마감일은 2024-12-31입니다."
print(extract_entities(test_query))

# Job search tool integration
@tool
def search_jobs(query: str) -> str:
    lang = detect_language(query)
    if lang not in vectorstores:
        data = load_data_from_file(lang)
        if data is None:
            data = jobploy_crawler(lang)
            save_data_to_file(data, lang)
        vectorstores[lang] = create_vectorstore(data)

    entities = extract_entities(query)

    location_filter = None
    if entities["LOCATION"]:
        location_filter = lambda d: any(
            loc.lower() in d.get('location', '').lower()
            for loc in entities["LOCATION"]
        )

    docs = vectorstores[lang].similarity_search(query, k=10)
    
    filtered_results = []

    for doc in docs:
        job_info = json.loads(doc.page_content)
        if location_filter and not location_filter(job_info):
            continue

        match = True

        if entities["MONEY"]:
            try:
                required_salary_str = entities["MONEY"][0].replace(',', '').replace('원', '').strip()
                required_salary = int(required_salary_str)
                
                pay_elements = job_info.get('pay', '').split()
                if len(pay_elements) >= 3:
                    job_salary_str = pay_elements[2].replace(',', '').replace('원', '').strip()
                    job_salary = int(job_salary_str)    
                    
                    if job_salary < required_salary:
                        continue
                else:
                    continue
                    
            except ValueError:
                continue

        if entities["OCCUPATION"] and match:
            occupation_match = any(
                occ.lower() in job_info.get('title', '').lower() or 
                occ.lower() in job_info.get('task', '').lower() 
                for occ in entities["OCCUPATION"]
            )
            if not occupation_match:
                match = False

        if match:
            filtered_results.append(job_info)

    if not filtered_results:
        return "No job listings found for the specified criteria."

    return json.dumps({
        "search_summary": {
            "total_jobs_found": len(filtered_results)
        },
        "job_listings": filtered_results,
        "additional_info": f"These are the job listings that match your query."
    }, ensure_ascii=False, indent=2)

# Vector store initialization and job crawler
def jobploy_crawler(lang, pages=3):
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []

    try:
        for page in range(1, pages + 1):
            url = f"https://www.jobploy.kr/{lang}/recruit?page={page}"
            driver.get(url)

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

            for job in job_listings:
                title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
                link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                pay_element = job.find_element(By.CSS_SELECTOR, "p.pay")
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")

                if len(badge_elements) >= 3:
                    location_element = badge_elements[0]
                    task_element = badge_elements[1]
                    closing_date_element = badge_elements[2]
                    location = location_element.text
                    task = task_element.text
                    closing_date = closing_date_element.text
                else:
                    closing_date = "마감 정보 없음"
                    location = "위치 정보 없음"
                    task = "직무 정보 없음"

                title = title_element.text
                pay = pay_element.text
                
                results.append({
                    "title": title,
                    "link": link_element,
                    "closing_date": closing_date,
                    "location": location,
                    "pay": pay,
                    "task": task,
                    "language": lang
                })

    finally:
        driver.quit()
        
    return results

def save_data_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def load_data_from_file(lang):
    filename = f"crawled_data_{lang}.txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None

def create_vectorstore(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents([
        Document(
            page_content=json.dumps(item, ensure_ascii=False),
            metadata={"location": item.get("location", "")}
        ) 
        for item in data
    ])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts([text.page_content for text in texts], embeddings, metadatas=[text.metadata for text in texts])

# Initialize vector stores
vectorstores = {}

tools = [search_jobs]

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a specialized AI assistant focused on job searches in Korea. Your primary function is to provide accurate, relevant, and up-to-date job information based on user queries.

            You utilize the following NER categories to extract key information from user queries:
            - LOCATION: Identifies geographical locations (e.g., cities, provinces)
            - MONEY: Recognizes salary or wage information
            - OCCUPATION: Detects job titles or professions
            - CLOSINGDATE: Identifies job application deadlines

            When responding to queries:
            1. Analyze user queries by language type to extract relevant entities.
            2. Search the job database using the extracted information.
            3. Filter and prioritize job listings based on the user's requirements.
            4. If the query includes a Location keyword, ensure that all relevant job listings for that location are retrieved and included in the response.
            5. Provide a comprehensive summary of the search results.
            6. Offer detailed information about each relevant job listing.
            7. Filter job listings based on the type of salary information requested by the user.
            8. If the keyword or numerical value does not match the user's query, do not provide any other data.

            Include the following information for each job listing:
            - Title
            - Company (if available)
            - Location
            - Task
            - Salary information
            - Closing date
            - Brief job description (if available)
            - Key requirements (if available)
            - Application link

            Ensure your response is clear, concise, and directly addresses the user's query. If a LOCATION is mentioned in the query, include all relevant job listings from that location.
            """
        ),
        (
            "user",
            "I need you to search for job listings based on my query. Can you help me with that?"
        ),
        (
            "assistant",
            "Certainly! I'd be happy to help you search for job listings based on your query. Please provide me with your specific request, and I'll use the search_jobs tool to find relevant information for you. What kind of job or criteria are you looking for?"
        ),
        (
            "user",
            "{input}"
        ),
        (
            "assistant",
            "Thank you for your query. I'll search for job listings based on your request using the search_jobs tool. I'll provide a summary of the results and detailed information about relevant job listings."
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/search_jobs")
async def search_jobs_endpoint(request: Request, query: str = Form(...)):
    chat_history = []

    detected_lang = detect_language(query)
    if detected_lang not in vectorstores:
        data = load_data_from_file(detected_lang)
        if data is None:
            data = jobploy_crawler(detected_lang)
            save_data_to_file(data, detected_lang)
        vectorstores[detected_lang] = create_vectorstore(data)

    # Fine-tune the model based on detected language
    fine_tune_model_for_language(detected_lang)

    result = agent_executor.invoke({"input": query, "chat_history": chat_history})
    chat_history.extend(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["output"]},
        ]
    )

    return JSONResponse(content={"response": result["output"], "chat_history": chat_history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)