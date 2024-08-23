# app/routes/items.py

from app.models.study_analysis import text_to_speech, study_analysis
from app.models.law_and_visa.law_and_visa_main import process_law_request, ChatRequest
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from pydantic import BaseModel
from app.models.medical import MedicalAssistant
from typing import Optional, Dict
from sqlalchemy.orm import Session
from app.database.db import get_db
from app.database import crud
import asyncio
import json
from fastapi.responses import StreamingResponse
import uuid

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    userNo: int
    categoryNo: int
    session_id: Optional[str] = None
    chat_his_no: Optional[int] = None
    is_new_session: Optional[bool] = None

class TextRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    chatHisNo: int
    chatHisSeq: int
    detected_language: str

medical_assistant = MedicalAssistant()

# 세션 ID와 chat_his_no 매핑을 위한 딕셔너리
session_chat_mapping: Dict[str, int] = {}

@router.post("/medical", response_model=ChatResponse)
async def medical_endpoint(request: Request, chat_request: ChatRequest, db: Session = Depends(get_db)):
    try:
        question = chat_request.question
        userNo = chat_request.userNo
        categoryNo = chat_request.categoryNo
        session_id = chat_request.session_id or str(uuid.uuid4())
        chat_his_no = chat_request.chat_his_no

        result = await medical_assistant.provide_medical_information(chat_request)

        is_new_session = chat_his_no is None and session_id not in session_chat_mapping
        current_chat_his_no = chat_his_no or session_chat_mapping.get(session_id)

        chat_history = crud.create_chat_history(
            db, 
            userNo, 
            categoryNo, 
            question, 
            result['answer'], 
            is_new_session=is_new_session, 
            chat_his_no=current_chat_his_no
        )
        
        session_chat_mapping[session_id] = chat_history.CHAT_HIS_NO

        async def generate_response():
            paragraphs = result['answer'].split('\n\n')
            for paragraph in paragraphs:
                words = paragraph.split()
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'content', 'text': word})}\n\n"
                    if i < len(words) - 1:
                        yield f"data: {json.dumps({'type': 'content', 'text': ' '})}\n\n"
                    await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'type': 'newline'})}\n\n"
                await asyncio.sleep(0.2)
            
            yield f"data: {json.dumps({'type': 'end', 'chatHisNo': chat_history.CHAT_HIS_NO, 'chatHisSeq': chat_history.CHAT_HIS_SEQ, 'detected_language': result['detected_language']})}\n\n"

        return StreamingResponse(generate_response(), media_type="text/event-stream")

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
    
@router.post("/law")
async def law_endpoint(chat_request: ChatRequest, db: Session = Depends(get_db)):
    return await process_law_request(chat_request, db)
