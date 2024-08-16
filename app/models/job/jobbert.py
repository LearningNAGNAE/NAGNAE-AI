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
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
import numpy as np

# Load environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 허용할 Origin 목록
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드 목록
    allow_headers=["*"],  # 허용할 HTTP 헤더 목록
)
templates = Jinja2Templates(directory="templates")  # html 파일내 동적 콘텐츠 삽입 할 수 있게 해줌(렌더링).

# Load tokenizer and model for BERT
model_name = "bert-base-multilingual-cased"
num_labels_dataset = 7  # 데이터셋의 레이블 수

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model config and update num_labels
config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels_dataset

# Load model with updated config
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

# NER pipeline setup
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

llm = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1
)

# Load and preprocess WikiANN dataset for fine-tuning
languages = ["ko", "en", "vi", "mn", "uz"]

# WikiANN 데이터셋 로드
datasets = {lang: load_dataset("wikiann", lang) for lang in languages}


# 데이터셋 확인 불러온 데이터에 몇 개의 예시(데이터)가 있는지 확인
for lang, dataset in datasets.items():
    print(f"{lang}: {len(dataset['train'])} training examples")

#단어를 토큰으로 나누고(자르기) 각 토큰에 맞는 라벨(태그)을 정리. 라벨이 필요 없는 부분은 -100이라는 값을 넣어 무시
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
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
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Preprocess datasets
tokenized_datasets = {lang: dataset.map(tokenize_and_align_labels, batched=True) for lang, dataset in datasets.items()}


# 모델 출력 크기 확인
num_labels_model = model.config.num_labels
print(f"Number of labels the model is expecting: {num_labels_model}")


# Combine datasets
combined_dataset = {
    "train": concatenate_datasets([dataset["train"] for dataset in tokenized_datasets.values()]),
    "validation": concatenate_datasets([dataset["validation"] for dataset in tokenized_datasets.values()]),
    "test": concatenate_datasets([dataset["test"] for dataset in tokenized_datasets.values()])
}

# 데이터셋 레이블 확인
num_labels_dataset = tokenized_datasets["ko"]["train"].features["ner_tags"].feature.num_classes
print(f"Number of labels in the dataset: {num_labels_dataset}")

# 두 값이 일치하는지 확인
if num_labels_model != num_labels_dataset:
    print("Warning: The number of labels in the model and dataset do not match!")
else:
    print("The number of labels in the model and dataset match.")
# **New Section: Extract unique labels from the dataset and create a label mapping**
unique_labels = set()
for example in combined_dataset["train"]:  # **Updated to use combined_dataset**
    unique_labels.update(example["labels"])

print(f"Unique labels in the dataset: {unique_labels}")

# Create a mapping from label to index
label_list = list(unique_labels)
label_to_index = {label: idx for idx, label in enumerate(label_list)}

# **New Section: Function to preprocess and map labels**
def preprocess_function(examples):
    examples['labels'] = [label_to_index[label] for label in examples['labels']]
    return examples

# **Apply label mapping to datasets**
tokenized_datasets = {lang: dataset.map(preprocess_function) for lang, dataset in tokenized_datasets.items()}  # **Updated to apply preprocessing**


# Define metrics for evaluation
metric = load_metric("seqeval", trust_remote_code=True)


#모델이 얼마나 잘했는지 평가하기 위해 평가 기준
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    #진짜 예측: 모델이 예측한 값을 라벨과 비교하여 맞았는지 확인
    true_predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    #정밀도, 재현율, F1 점수, 정확도
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Fine-tuning setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=4e-4, #(con)
    per_device_train_batch_size=30,#(con)
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=1, #학습 반복 횟수 #(con)
    weight_decay=0.01, #가중치 감쇠
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset["train"],
    eval_dataset=combined_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluation
results = trainer.evaluate()
print(results)

# Language detection function
def detect_language(text):
    lang, _ = classify(text)
    return lang

# Entity extraction function using the fine-tuned model
def extract_entities(query):
    print("Using the fine-tuned BERT model to extract entities.")
    
    ner_results = ner_pipeline(query)
    
    print("NER results:", ner_results)

    entities = {
        "LOCATION": [],
        "MONEY": [],
        "OCCUPATION": []
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
                if current_label in ["LOC", "LABEL_0"]:
                    entities["LOCATION"].append(current_entity)
                elif current_label in ["MONEY", "LABEL_1"]:
                    entities["MONEY"].append(current_entity)
                elif current_label in ["OCCUPATION", "LABEL_2"]:
                    entities["OCCUPATION"].append(current_entity)
            current_entity = entity_text
            current_label = entity_label
    
    if current_entity:
        if current_label in ["LOC", "LABEL_0"]:
            entities["LOCATION"].append(current_entity)
        elif current_label in ["MONEY", "LABEL_1"]:
            entities["MONEY"].append(current_entity)
        elif current_label in ["OCCUPATION", "LABEL_2"]:
            entities["OCCUPATION"].append(current_entity)
    
    money_pattern = re.compile(r'\d{1,3}(,\d{3})*(원|KRW)?')
    matches = money_pattern.findall(query)
    for match in matches:
        entities["MONEY"].append(match[0])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'job_location.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        job_locations = json.load(f)

    words = query.split()
    for word in words:
        for location, data in job_locations.items():
            if word in data['aliases']:
                entities["LOCATION"].append(location)
                break
            for area in data['areas']:
                if word == area['ko'] or word == area['en']:
                    entities["LOCATION"].append(f"{location} {area['ko']}")
                    break
            else:
                continue
            break

    print("Extracted entities:", entities)
    return entities

# Sample query for testing
test_query = "경기도 화성시에서 300만원 이상의 월급을 받는 일을 찾고 있습니다."
print(extract_entities(test_query))

# Job search tool integration
@tool
def search_jobs(query: str) -> str:
    """Search for job listings based on the given query."""
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
    if isinstance(pages, dict):
        pages = 3
    elif not isinstance(pages, int):
        try:
            pages = int(pages)
        except ValueError:
            pages = 3

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

default_lang = 'ko'
crawled_data = jobploy_crawler(lang=default_lang, pages=3)
save_data_to_file(crawled_data, f"crawled_data_{default_lang}.txt")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents([
    Document(page_content=json.dumps(item, ensure_ascii=False)) 
    for item in crawled_data
])

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