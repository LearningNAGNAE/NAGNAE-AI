from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn

# Selenium Web Scraper
def jobploy_crawler(pages=5):
    chrome_driver_path = r"C:\chromedriver\chromedriver.exe"  # Update to the actual path

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920x1080")  # Set window size

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []
    try:
        for page in range(1, pages+1):
            driver.get(f"https://www.jobploy.kr/ko/recruit?page={page}")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content"))
            )

            job_listings = driver.find_elements(By.CSS_SELECTOR, ".item.col-6")

            for job in job_listings:
                title_element = job.find_element(By.CSS_SELECTOR, "h6.mb-1")
                link_element = job.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                location_element = job.find_element(By.CSS_SELECTOR, ".item_link")
                salary_element = job.find_element(By.CSS_SELECTOR, "p.pay")

                # 모든 배지 요소를 가져옴
                badge_elements = job.find_elements(By.CSS_SELECTOR, ".badge.text-dark.bg-secondary-150.rounded-pill")
                
                # 배지 정보 추출
                location = badge_elements[0].text if len(badge_elements) >= 1 else "지역 정보 없음"
                job_role = badge_elements[1].text if len(badge_elements) >= 2 else "직무 정보 없음"
                closing_date = badge_elements[2].text if len(badge_elements) >= 3 else "마감 정보 없음"

                title = title_element.text
                salary = salary_element.text

                results.append({
                    "title": title,
                    "link": link_element,
                    "location": location,
                    "job_role": job_role,
                    "salary": salary,
                    "closing_date": closing_date
                })

    finally:
        driver.quit()
    
    return results

# Initial Data Preparation
results = jobploy_crawler(pages=5)
documents = [{"text": result['title'], "metadata": {"link": result['link'], "location": result['location'], "job_role": result['job_role'], "salary": result['salary'], "closing_date": result['closing_date']}} for result in results]

# Initialize SentenceTransformer and FAISS
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([doc["text"] for doc in documents])

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Map embeddings to their metadata
id_to_metadata = {i: doc["metadata"] for i, doc in enumerate(documents)}

# FastAPI app setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
def query_jobs(request: QueryRequest):
    query = request.query

    # Embed the query
    query_embedding = model.encode([query])

    # Search in the FAISS index
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 results

    if len(I[0]) == 0:
        raise HTTPException(status_code=404, detail="No matching jobs found")

    # Generate the response
    response = {
        "query": query,
        "results": []
    }

    for i in I[0]:
        metadata = id_to_metadata.get(i, {})
        document_text = documents[i].get("text", "정보 없음")
        
        # 디버깅 출력
        print(f"Metadata: {metadata}")
        print(f"Document: {documents[i]}")
        
        response["results"].append({
            "title": document_text,
            "link": metadata.get("link", "정보 없음"),
            "location": metadata.get("location", "정보 없음"),
            "job_role": metadata.get("job_role", "정보 없음"),
            "salary": metadata.get("salary", "정보 없음"),
            "closing_date": metadata.get("closing_date", "정보 없음")
        })

    return response


# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
