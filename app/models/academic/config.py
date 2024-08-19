import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from elasticsearch import Elasticsearch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# 환경 변수 로드
university_api_key = os.getenv("UNIVERSITY_API_KEY")
gpt_api_key = os.getenv("OPENAI_API_KEY")
elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
pdf_path = r"C:\Users\hi02\dev\NAGNAE\NAGNAE-AI\pdf\외국인 전형 대학 정보.pdf"

# CrossEncoder 모델 초기화
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')

# OpenAI 모델 초기화
openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=gpt_api_key, temperature=0)

# Elasticsearch 클라이언트 설정
es_client = Elasticsearch([elasticsearch_url])
embedding = OpenAIEmbeddings()

# 세션별 대화 기록을 저장할 딕셔너리
session_histories: Dict[str, List] = {}

# 배치 크기와 요청 간 대기 시간 설정
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 1  # 초 단위

# 파인튜닝된 모델 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fine_tuned_qwen2_1_5b")

# 파인튜닝된 모델과 토크나이저 로드
fine_tuned_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to('cuda')
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=True)
