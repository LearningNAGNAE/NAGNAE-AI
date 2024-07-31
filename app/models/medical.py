import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 현재 파일의 디렉토리 경로를 가져옴
current_dir = os.path.dirname(os.path.abspath(__file__))
# medical-faiss 디렉토리 경로를 구성
medical_faiss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "medical", "medical_faiss")

class Medical:

    @classmethod
    def format_docs(cls, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    @classmethod
    def load_faiss_index(cls):
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        load_db = FAISS.load_local(medical_faiss_dir, embeddings, allow_dangerous_deserialization=True)
        return load_db

    @classmethod
    async def chatbot(cls, query: str):
        load_dotenv()
        db = cls.load_faiss_index()

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # 사용자 정의 프롬프트 템플릿 생성
        prompt_template = """
        당신은 외국인 대상 의료관련에 대해서 대답해주는 유능한 AI 비서입니다. 주어진 맥락 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해야 합니다.

        맥락: {context}

        질문: {question}

        답변을 작성할 때 다음 지침을 따르세요:
        1. 주어진 맥락 정보에 있는 내용만을 사용하여 답변하세요.
        2. 맥락 정보에 없는 내용은 답변에 포함하지 마세요.
        3. 질문과 관련이 없는 정보는 제외하세요.
        4. 답변은 간결하고 명확하게 작성하세요.
        5. 불확실한 경우, "주어진 정보로는 정확한 답변을 드릴 수 없습니다."라고 말하세요.

        답변:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = (
            {
                "context": db.as_retriever() | cls.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 질문에 대한 답변 생성
        result = qa_chain.invoke(query)
        return result
