import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.models.law_and_visa.law_and_visa_util import setup_driver
from app.models.law_and_visa.law_and_visa_embedding import scrap_law_a, scrap_law_b

load_dotenv()

def process_and_embed(laws, embeddings, index_name):
    documents = []
    for law in laws:
        content = f"제목: {law['title']}\n\n"
        if isinstance(law['content'], dict):
            for section, section_content in law['content'].items():
                content += f"{section}:\n{section_content}\n\n"
        else:
            content += f"내용:\n{law['content']}\n\n"
        documents.append(Document(page_content=content, metadata={"title": law['title']}))
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(texts, embeddings)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f"faiss_index_{index_name}")
    
    try:
        vector_store.save_local(save_path)
        print(f"FAISS 인덱스가 성공적으로 저장되었습니다: {save_path}")
    except Exception as e:
        print(f"FAISS 인덱스 저장 중 오류 발생: {e}")
    
    if os.path.exists(save_path):
        print(f"FAISS 인덱스 파일 확인: {save_path}")
    else:
        print(f"FAISS 인덱스 파일이 존재하지 않습니다: {save_path}")
    
    return len(texts)

def main():
    #a.종합법률정보: 판례
    scrap_law_a_url = "https://glaw.scourt.go.kr/wsjo/panre/sjo060.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#1721785596647"

    #b.종합법률정보: 법령
    scrap_law_b_url = "https://glaw.scourt.go.kr/wsjo/lawod/sjo130.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p4=02#//"

    #c.종합법률정보: 조약
    scrap_law_c_url = "https://glaw.scourt.go.kr/wsjo/trty/sjo610.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8#//"

    #d.종합법률정보: 규칙/예규/선례
    scrap_law_d_url = "https://glaw.scourt.go.kr/wsjo/gchick/sjo300.do?prevUrl=interSrch&tabId=0&q=%EC%99%B8%EA%B5%AD%EC%9D%B8&p2=01#//"

    driver = setup_driver()

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # future_a = executor.submit(scrap_law_a, driver, scrap_law_a_url)
            # future_b = executor.submit(scrap_law_b, driver, scrap_law_b_url)
            # future_c = executor.submit(scrap_law_b, driver, scrap_law_c_url)  # scrap_law_b 함수를 재사용
            future_d = executor.submit(scrap_law_b, driver, scrap_law_d_url)  # scrap_law_b 함수를 재사용

            # laws_a = future_a.result()
            # laws_b = future_b.result()
            # laws_c = future_c.result()
            laws_d = future_d.result()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # future_a = executor.submit(process_and_embed, laws_a, embeddings, "panre")
            # future_b = executor.submit(process_and_embed, laws_b, embeddings, "lawod")
            # future_c = executor.submit(process_and_embed, laws_c, embeddings, "trty")
            future_d = executor.submit(process_and_embed, laws_d, embeddings, "gchick")

            # count_a = future_a.result()
            # count_b = future_b.result()
            # count_c = future_c.result()
            count_d = future_d.result()

        # print(f"임베딩 완료: 판례 {count_a}개")
        # print(f"임베딩 완료: 법령 {count_b}개")
        # print(f"임베딩 완료: 조약 {count_c}개")
        print(f"임베딩 완료: 규칙/예규/선례 {count_d}개")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        driver.quit()

    print("전체 프로세스 완료")

if __name__ == "__main__":
    main()