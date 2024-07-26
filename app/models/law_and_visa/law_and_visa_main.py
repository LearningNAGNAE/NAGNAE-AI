from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langdetect import detect
from googletrans import Translator

load_dotenv()

def detect_language(text):
    return detect(text)

def translate_text(text, target_lang='ko'):
    translator = Translator()
    return translator.translate(text, dest=target_lang).text

def main():
    embeddings = OpenAIEmbeddings()
    collection_name = "law_and_visa"
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": "localhost", "port": "19530"}
    )

    llm = OpenAI(temperature=0)

    system_prompt = """
    당신은 한국의 외국인 노동자와 유학생을 위한 법률 상담 전문가이자 통역/번역가입니다. 
    주어진 정보를 바탕으로 친절하고 명확하게 설명해주세요. 
    법률 정보를 제공할 때는 정확성을 유지하면서도 이해하기 쉽게 설명해야 합니다.
    답변할 수 없는 질문에 대해서는 솔직히 모른다고 인정하고, 필요한 경우 전문가와 상담을 권유하세요.
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt + """

    관련 정보: {context}

    질문: {question}

    답변:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )

    while True:
        query = input("질문을 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
        
        original_lang = detect_language(query)
        if original_lang != 'ko':
            query_ko = translate_text(query, 'ko')
        else:
            query_ko = query

        result_ko = qa_chain.run(query_ko)

        if original_lang != 'ko':
            result = translate_text(result_ko, original_lang)
        else:
            result = result_ko

        print(f"답변: {result}\n")

    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()