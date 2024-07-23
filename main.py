# /main.py

from app.main import app  # app 패키지의 main 모듈에서 FastAPI app을 가져옵니다.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
