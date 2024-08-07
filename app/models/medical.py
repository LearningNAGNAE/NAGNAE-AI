import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from typing import List
from googleapiclient.discovery import build
from langchain.schema import BaseRetriever, Document

class GoogleSearchRetriever(BaseRetriever):

    def _search(self, query: str, **kwargs) -> List[Document]:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_CSE_ID"), num=5, dateRestrict="m6", fields="items(title,link,snippet)", **kwargs).execute()
        documents = []

        for item in res.get('items', []):
            doc = Document(
                page_content=item['snippet'],
                metadata={'source': item['link'], 'title': item['title']}
            )
            documents.append(doc)
    
        return documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._search(query)

class MedicalAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = self.__initialize_faiss_index()
        self.ensemble_retriever = self.__setup_ensemble_retriever()
        # self.google_retriever = GoogleSearchRetriever()

    def __initialize_faiss_index(self):
        medical_faiss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "medical", "medical_faiss")
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        return FAISS.load_local(medical_faiss_dir, embeddings, allow_dangerous_deserialization=True)

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

    def __setup_ensemble_retriever(self):
        bm25_texts = [
            "Emergency medical services: Call 119 for immediate assistance. English support available. Check KDCA website for current health advisories.",
            "Health insurance: Mandatory for most visas. Maintain coverage when changing jobs. Covers 60-80% of medical expenses. Visit www.nhis.or.kr for details.",
            "Hospitals for foreigners: Seoul - Severance, Samsung Medical Center. Busan - Pusan National University Hospital. Many offer language services.",
            "Preventive care: Free annual check-ups for National Health Insurance subscribers over 40. Includes blood tests, chest X-ray, and cancer screenings.",
            "Medical information resources: NHIS (1577-1000) for insurance. HIRA (1644-2000) for medical costs. Medical 1339 for 24/7 health advice in English.",
            "Common medications: Tylenol (acetaminophen) for pain/fever, available OTC. Prescription needed for antibiotics. Always consult a pharmacist.",
            "Mental health services: Seoul Global Center offers free counseling. National suicide prevention hotline: 1393 (24/7, English available).",
            "Vaccinations: MMR, influenza, and hepatitis B recommended. Many available for free or discounted rates for insurance subscribers.",
            "COVID-19 information: Check KDCA website for current guidelines. Free testing and treatment for confirmed cases. Mask required in medical facilities.",
            "Specialist care: Referral from general practitioner often required. International clinics at major hospitals can assist with appointments.",
        ]
        bm25_retriever = BM25Retriever.from_texts(bm25_texts, k=2)
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        google_retriever = GoogleSearchRetriever()

        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever, google_retriever],
            weights=[0.2, 0.5, 0.3]
        )

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

    async def provide_medical_information(self, query: str):
        print(f"Received question: {query}")
        language = self.__identify_query_language(query)

        if self.ensemble_retriever is None:
            error_message = "System error. Please try again later."
            return {
                "question": query,
                "answer": error_message,
                "detected_language": language,
                "google_search_results": []
            }

        retrieval_chain = self.__create_retrieval_chain()
        response = retrieval_chain.invoke({"question": query, "language": language})

        # Google Custom Search API 결과 가져오기
        # google_search_results = self.google_retriever.get_relevant_documents(query)

        return {
            "question": query,
            "answer": response,
            "detected_language": language
            # "detected_language": language,
            # "google_search_results": [
            #     {
            #         "title": doc.metadata.get('title', ''),
            #         "link": doc.metadata.get('source', ''),
            #         "snippet": doc.page_content
            #     } for doc in google_search_results
            # ]
        }