import os
from typing import List
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class GoogleSearchRetriever:
    def __init__(self):
        self._wrapper = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID")
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self._wrapper.results(query, num_results=2)
        return [
            Document(
                page_content=result.get("snippet", "No description available"),
                metadata={'source': result.get('link', ''), 'title': result.get('title', '')}
            )
            for result in results
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# 테스트 코드
if __name__ == "__main__":
    retriever = GoogleSearchRetriever()
    docs = retriever.get_relevant_documents("example query")
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)