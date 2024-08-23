# app/routes/items.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from app.models.medical import MedicalAssistant
from app.models.study_analysis import text_to_speech, study_analysis, study_image_analysis
from app.models.law_and_visa.law_and_visa_main import process_law_request, ChatRequest
from typing import Dict, Any
from sqlalchemy.orm import Session
from app.database.db import get_db

router = APIRouter()


class Query(BaseModel):
    input: str

class TextRequest(BaseModel):
    text: str

medical_assistant = MedicalAssistant()

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

@router.post("/study-analysis")
async def study_analysis_endpoint(file: UploadFile = File(..., max_size=1024*1024*10)):
    try:
        file_content = await file.read()
        result = await study_analysis(file_content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/study-image-analysis")
async def study_image_analysis_endpoint(
    audio_file: UploadFile = File(..., description="The audio file to be analyzed"),
    image_file: UploadFile = File(..., description="The image file to be analyzed")
) -> Dict[str, Any]:
    try:
        audio_content = await audio_file.read()
        image_content = await image_file.read()
        result = await study_image_analysis(audio_content, image_content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/law")
async def law_endpoint(chat_request: ChatRequest, db: Session = Depends(get_db)):
    return await process_law_request(chat_request, db)
