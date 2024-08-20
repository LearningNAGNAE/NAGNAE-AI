# app/routes/items.py

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from app.models.grammar_correction import grammar_corrector
from app.models.t2t import t2t
from pydantic import BaseModel
from app.models.academic.study_crawl import setup_langchain
from contextlib import asynccontextmanager
from app.models.medical import MedicalAssistant
from app.models.job_crawl import create_agent_executor_job
from app.models.study_analysis import text_to_speech
from app.models.study_analysis import study_text_analysis
from app.models.study_analysis import study_image_analysis
from typing import Dict, Any

router = APIRouter()

class Query(BaseModel):
    input: str

class TextRequest(BaseModel):
    text: str

agent_executor = None
medical_assistant = MedicalAssistant()

@router.post("/job_search")
async def job_search(query: Query, request: Request):
    agent_executor_job = request.app.state.agent_executor_job
    if agent_executor_job is None:
        raise HTTPException(status_code=500, detail="Agent executor not initialized")
    
    try:
        result = await agent_executor_job.ainvoke({"input": query.input})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app):
    # 시작 시 실행할 코드
    global agent_executor
    agent_executor = setup_langchain()
    app.state.agent_executor_job = await create_agent_executor_job()
    yield
    # 종료 시 실행할 코드 (필요한 경우)

@router.post("/study")
def query_agent(query: Query):
    try:
        agent_result = agent_executor.invoke({"input": query.input})
        return {"result": agent_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/grammar-correct")
async def grammar_correct_endpoint(file: UploadFile = File(...)):
    try:
        # 텍스트 파일 비동기적으로 읽기
        text_data = (await file.read()).decode("utf-8").strip()

        # grammar_corrector.correct_text 호출
        response = await grammar_corrector.correct_text(text_data)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/t2t")
async def t2t_endpoint(file: UploadFile = File(...)):
    try:
        # 텍스트 파일 비동기적으로 읽기
        text_data = (await file.read()).decode("utf-8").strip()

        # t2t.generate_code 호출
        result = await t2t.generate_code(text_data)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/medical")
async def medical_endpoint(query: Query):
    try:
        result = await medical_assistant.provide_medical_information(query.input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@router.post("/text-to-speech")
async def text_to_speech_endpoint(request: TextRequest):
    try:
        return await text_to_speech(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/study-text-analysis")
async def study_text_analysis_endpoint(file: UploadFile = File(..., max_size=1024*1024*10)):
    try:
        file_content = await file.read()
        result = await study_text_analysis(file_content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/study-image-analysis")
async def study_image_analysis_endpoint(
    audio_file: UploadFile = File(..., description="The audio file to be analyzed"),
    image_file: UploadFile = File(..., description="The image file to be analyzed")
) -> Dict[str, Any]:
    try:
        # 오디오 파일 처리
        audio_content = await audio_file.read()
        image_content = await image_file.read()
        result = await study_image_analysis(audio_content, image_content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))