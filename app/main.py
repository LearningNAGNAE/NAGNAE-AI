# /app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import items  # items 라우터를 임포트합니다.
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(lifespan=items.lifespan)

# CORS 설정
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# items 라우터를 등록합니다.
app.include_router(items.router)

