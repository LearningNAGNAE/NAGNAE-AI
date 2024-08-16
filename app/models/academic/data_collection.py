import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import university_api_key, pdf_path
from langchain.document_loaders import PyPDFLoader

def fetch_university_data(per_page=30000):
    """대학 정보 데이터 수집"""
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={university_api_key}&svcType=api&svcCode=SCHOOL&contentType=json&gubun=univ_list&thisPage=1&perPage={per_page}"
    response = requests.get(url)
    return response.json()

def fetch_university_major(per_page=30000):
    """대학 전공 데이터 수집"""
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={university_api_key}&svcType=api&svcCode=MAJOR&contentType=json&gubun=univ_list&thisPage=1&perPage={per_page}"
    response = requests.get(url)
    return response.json()

def fetch_major_details(major_seq):
    """전공 상세 정보 데이터 수집"""
    url = f"https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey={university_api_key}&svcType=api&svcCode=MAJOR_VIEW&contentType=json&gubun=univ_list&majorSeq={major_seq}"
    response = requests.get(url)
    return response.json()

def load_pdf_document(pdf_path):
    
    """PDF 파일의 데이터 수집"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents