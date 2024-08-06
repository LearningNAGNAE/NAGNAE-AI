import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class MedicalAssistant:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = self.__initialize_faiss_index()
        self.ensemble_retriever = self.__setup_ensemble_retriever()

    def __concatenate_document_contents(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
            "Korean labor laws protect workers' rights to fair wages and working conditions.",
            "The visa application process in Korea involves several steps and documentation.",
            # 추가 텍스트 데이터 ...
        ]
        bm25_retriever = BM25Retriever.from_texts(bm25_texts, k=4)
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])

    def __create_chat_prompt(self):
        system_prompt = """
        # AI Assistant for Foreign Workers and Students in Korea: Medical and General information

        ## Role and Responsibility
        You are a specialized AI assistant providing information on Medical information, Health consultation services and general topics for foreign workers and students in Korea. Your primary goals are to:

        1. Provide accurate, easy-to-understand information in the language specified in the 'RESPONSE_LANGUAGE' field.
        2. Offer both general and specific information relevant to foreign workers and students.
        3. Guide users on medical information and health consultation.
        4. Ensure cultural sensitivity and awareness in all interactions.
        5. Provide relevant contact information or website addresses for official authorities when applicable.

        ## Guidelines

        1. Language: ALWAYS respond in the language specified in the 'RESPONSE_LANGUAGE' field. This will match the user's question language.

        2. Information Scope:
        - Healthcare System: Korean medical system, how to use hospitals, emergency medical services
        - Health Insurance: National Health Insurance, private insurance, insurance enrollment procedures
        - Disease Prevention: Vaccinations, health screenings, hygiene management
        - Mental Health: Counseling services, stress management, cultural adaptation

        3. Specific Focus Areas:
        - Differences between medical services for foreigners and general medical services
        - Guidance on medical institutions with language support services
        - Information on medical dispute resolution and official channels for help

        4. Cultural Sensitivity: Be aware of cultural differences and provide necessary context.

        5. Uncertainty Handling: If uncertain, respond in the specified language with:
        "Based on the provided information, I cannot give a definitive answer. Please consult [relevant medical institution] or a medical professional specializing in [specific area] for accurate advice."

        6. Medical Disclaimers: Emphasize that medical information is for general guidance only and cannot replace diagnosis by a medical professional.

        7. Contact Information: When relevant, provide official contact information or website addresses for appropriate government agencies or organizations. For example:
        - National Health Insurance Service: www.nhis.or.kr
        - Korea Disease Control and Prevention Agency: www.kdca.go.kr
        - Emergency Medical Information Center: www.e-gen.or.kr
        - Korea Immigration Service: www.hikorea.go.kr

        Always approach each query systematically to ensure accurate, helpful, and responsible assistance. Prioritize the health and well-being of foreign workers and students in your responses, and guide them to professional medical personnel or official medical institutions when necessary.
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
            }

        retrieval_chain = self.__create_retrieval_chain()
        response = retrieval_chain.invoke({"question": query, "language": language})

        return {
            "question": query,
            "answer": response,
            "detected_language": language
        }