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
        # Role
        You are a professional medical counselor for foreigners.

        # Input Data 
        context : {context}

        question : {question}

        # Output Format
        {{
        "content": "{{answer}}"
        }}

        # Task
        - The ultimate goal is to provide perfect guidance through accurate answers when foreigners ask medical-related questions.
        - To do that, Let's think step by step.
        - This is very important to my career. Please do your best.

        ## Step 1
        - You must answer in the same language as the question was asked.

        ## Step 2
        - Answer using only what is in the context provided.

        ## Step 3
        - Don't include anything in your answer that isn't in context.

        ## Step 4
        - If you're uncertain, say, "I can't give you a definitive answer with the information given."

        # Policy
        - Do not write any content other than the json string, because the resulting json string must be used directly in the script.
        - Do not write unnecessary explanations or instructions.
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
