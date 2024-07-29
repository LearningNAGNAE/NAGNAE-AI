# app/routes/items.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.grammar_correction import grammar_corrector
from app.models.t2t import t2t
from app.models.medical import Medical

router = APIRouter()

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

@router.get("/medical")
async def medical(query: str):
    try:
        # 질문에 대한 답변 생성
        result = await Medical.chatbot(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))