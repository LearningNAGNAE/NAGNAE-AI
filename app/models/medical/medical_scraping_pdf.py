import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader

# 현재 스크립트의 디렉토리 경로를 얻습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(current_dir, "scraping_file")

def download_pdf_file(url, save_path):
    """PDF 파일을 다운로드하여 지정된 경로에 저장"""
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, 'wb') as file:
        file.write(response.content)

def get_file_link(page_url):
    """페이지에서 파일의 링크를 추출"""
    response = requests.get(page_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    file_link_tag = soup.find("a", {"class": "btn_line", "href": "/boardDownload.es?bid=0026&list_no=1481348&seq=2"})
    if not file_link_tag:
        raise ValueError("파일 링크를 찾을 수 없습니다")
    file_link = file_link_tag['href']
    return f"https://www.mohw.go.kr{file_link}"

def process_pdf(file_path):
    """PDF 파일을 로드하여 텍스트를 추출"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def main():

    page_url = "https://www.mohw.go.kr/board.es?mid=a10409020000&bid=0026&list_no=1481348&act=view&"
    pdf_link = get_file_link(page_url)
    pdf_save_path = os.path.join(save_directory, "long_term_stay_health_insurance_standards.pdf")
    download_pdf_file(pdf_link, pdf_save_path)
    documents = process_pdf(pdf_save_path)

    return documents