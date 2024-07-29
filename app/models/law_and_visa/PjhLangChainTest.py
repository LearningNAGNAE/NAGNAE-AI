from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.messages import stream_response  # 스트리밍 출력
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()

#LangSmith 추적 설정
# set_enable=False 로 지정하면 추적을 하지 않습니다.
logging.langsmith("NAGNAE")

class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)


# 스트리밍을 활성화하기 위해, ChatModel 생성자에 `streaming=True`를 전달합니다.
# 추가적으로, 사용자 정의 핸들러 리스트를 전달합니다.
stream_llm = ChatOpenAI(
    model="gpt-4o-mini", streaming=True, max_tokens=50, callbacks=[StreamingHandler()]
)

conversation = ConversationChain(
    llm=stream_llm,
    verbose=False,
    memory=ConversationBufferMemory(),
)


output = conversation.predict(input="양자역학에 대해 설명해줘")



# prompt = PromptTemplate.from_template("{topic} 에 대해 쉽게 설명해주세요.")

# # 객체 생성
# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     max_tokens=2048,
#     temperature=0.1, #창의성(0.0~2.0)
# )

# # chain = prompt | model

# output_parser = StrOutputParser()

# chain = prompt | model | output_parser
# # print("11wwwwww1")
# # input 딕셔너리에 주제를 '인공지능 모델의 학습 원리'으로 설정합니다.
# input = {"topic": "인공지능 모델의 학습 원리"}


# # 질의
# # 질의
# try:
#     response = chain.stream(input)
#     stream_response(response)
# except Exception as e:
#     print(f"Error: {e}")
# # print(f"[답변]: {response.content}")


