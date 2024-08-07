import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 필요한 모듈들을 임포트합니다.

class MedicalAssistant:
    def __init__(self):
        load_dotenv()  # .env 파일에서 환경 변수를 로드합니다.
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )  # OpenAI의 ChatGPT 모델을 초기화합니다.
        self.vector_store = self.__initialize_faiss_index()  # FAISS 인덱스를 초기화합니다.
        self.ensemble_retriever = self.__setup_ensemble_retriever()  # 앙상블 리트리버를 설정합니다.

    def __initialize_faiss_index(self):
        medical_faiss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "medical", "medical_faiss")
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        return FAISS.load_local(medical_faiss_dir, embeddings, allow_dangerous_deserialization=True)
        # FAISS 인덱스를 로컬에서 로드합니다.

    def __identify_query_language(self, text):
        system_prompt = "You are a language detection expert. Detect the language of the given text and respond with only the language name in English, using lowercase."
        human_prompt = f"Text: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ]
        response = self.llm.invoke(messages)
        detected_language = response.content.strip().lower()
        print(f"Detected language: {detected_language}")
        return detected_language
        # 쿼리 언어를 식별합니다.

    def __setup_ensemble_retriever(self):
        bm25_texts = [
            "Emergency medical services: Call 119 for immediate assistance. English support available. Check KDCA website for current health advisories.",
            "Health insurance: Mandatory for most visas. Maintain coverage when changing jobs. Covers most medical expenses at reduced rates.",
            "Hospitals for foreigners: Many offer language services. International Healthcare Centers available at major hospitals for specialized care.",
            "Preventive care: Free or discounted health check-ups and vaccinations for National Health Insurance subscribers. Regular screenings recommended.",
            "Medical information resources: Contact HIRA (1577-1000) or NHIS (1577-1000) for inquiries. Visit www.nhis.or.kr for comprehensive health insurance details."
        ]
        bm25_retriever = BM25Retriever.from_texts(bm25_texts, k=4)
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])
        # BM25와 FAISS 리트리버를 결합한 앙상블 리트리버를 설정합니다.

    def __create_chat_prompt(self):
        system_prompt = """
        # AI Assistant for Foreign Workers and Students in Korea: Detailed Medical and Healthcare Information

        ## Role and Responsibility
        You are a specialized AI assistant providing comprehensive information on the Korean healthcare system, medical services, health insurance, and general health-related topics for foreign workers and students in Korea. Your primary goals are to:

        1. Provide accurate, easy-to-understand information in the language specified in the 'RESPONSE_LANGUAGE' field.
        2. Offer both general and specific information relevant to foreign workers and students, with a focus on healthcare-related queries.
        3. Guide users on navigating the Korean healthcare system, insurance options, and accessing medical services.
        4. Ensure cultural sensitivity and awareness in all interactions.

        ## Guidelines

        1. Language: ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the user's question language.

        2. Information Scope:
        - Healthcare System: Provide detailed information on the Korean medical system, types of medical facilities, emergency services, and how to use them.
        - Health Insurance: Explain National Health Insurance and private insurance options, enrollment procedures, and coverage details.
        - Disease Prevention: Offer information on vaccinations, health screenings, and hygiene management in Korea.
        - Mental Health: Provide resources for counseling services, stress management, and cultural adaptation support.

        3. Specific Focus Areas:
        - Explain differences in medical services for foreigners compared to general services.
        - Provide guidance on medical institutions with language support services.
        - Detail the process for accessing healthcare services, including making appointments and paying for services.
        - Offer information on medical dispute resolution and official channels for assistance.
        
        4. Completeness: Always provide a comprehensive answer based on the available context. Include:
        - Step-by-step procedures for accessing healthcare services
        - Specific requirements for insurance enrollment or using medical services
        - Potential cultural differences in healthcare practices
        - Available support services for foreign patients

        5. Accuracy and Updates: Emphasize that while you provide detailed information, healthcare policies and services may change. Always advise users to verify current information with official sources.

        6. Structured Responses: Organize your responses clearly, using bullet points or numbered lists when appropriate to break down complex information.

        7. Examples and Scenarios: When relevant, provide examples or hypothetical scenarios to illustrate how the Korean healthcare system works in practice.

        8. Uncertainty Handling: If uncertain about specific details, clearly state this and provide the most relevant general information available. Always recommend consulting official sources or healthcare professionals for the most up-to-date and case-specific guidance.

        Remember, your goal is to provide as much relevant, accurate, and detailed healthcare information as possible while ensuring it's understandable and actionable for the user. Always prioritize the user's health and well-being, and guide them to professional medical personnel or official medical institutions when necessary.
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

        human_template = """
        RESPONSE_LANGUAGE: {language}
        CONTEXT: {context}
        QUESTION: {question}

        Please provide a detailed answer to the above question in the specified RESPONSE_LANGUAGE.
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        # 챗봇 프롬프트를 생성합니다.

    def __create_retrieval_chain(self):
        chat_prompt = self.__create_chat_prompt()
        return (
            {
                "context": lambda x: self.ensemble_retriever.get_relevant_documents(x["question"]),
                "question": RunnablePassthrough(),
                "language": lambda x: self.__identify_query_language(x["question"])
            }
            | chat_prompt
            | self.llm
            | StrOutputParser()
        )
        # 검색 체인을 생성합니다.

    async def provide_medical_information(self, query: str):
        print(f"Received question: {query}")
        language = self.__identify_query_language(query)

        if self.ensemble_retriever is None:
            error_message = "System error. Please try again later."
            return {
                "question": query,
                "answer": error_message,
                "detected_language": language,
            }

        retrieval_chain = self.__create_retrieval_chain()
        response = retrieval_chain.invoke({"question": query, "language": language})

        return {
            "question": query,
            "answer": response,
            "detected_language": language
        }
        # 의료 정보를 제공하는 비동기 메서드입니다.