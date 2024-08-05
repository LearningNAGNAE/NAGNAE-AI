import os
from medical_scraping_html import main as medical_scraping_html
from medical_scraping_pdf import main as medical_scraping_pdf
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(current_dir, "scraping_file")
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
medical_faiss_path = os.path.join(current_dir, "medical_faiss")

def create_directory(path):
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(path, exist_ok=True)

def split_documents(documents, chunk_size=300, chunk_overlap=50):
    """문서를 주어진 크기와 중첩으로 분할"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_faiss_vector_db(documents, embeddings, db_path):
    """문서에서 벡터 데이터베이스를 생성하고 저장"""
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)

def main():

    create_directory(save_directory)

    html_documents = split_documents(medical_scraping_html())
    pdf_documents = split_documents(medical_scraping_pdf())

    all_docs = html_documents + pdf_documents

    create_faiss_vector_db(all_docs, embeddings, medical_faiss_path)

if __name__ == "__main__":
    main()
