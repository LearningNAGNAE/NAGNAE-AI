import os
from happytransformer import HappyTextToText, TTSettings
from fastapi.responses import JSONResponse

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GrammarCorrector:
    happy_tt = None
    args = None

    @classmethod
    def load_model(cls):
        if cls.happy_tt is None:
            print("Loading HappyTextToText model...")
            cls.happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
            cls.args = TTSettings(num_beams=10, max_length=1000, min_length=1)
            print("HappyTextToText model loaded.")

    @classmethod
    async def correct_text(cls, text_data: str):
        try:
            cls.load_model()  # 필요할 때만 모델을 로드합니다.
            
            # 문법 교정
            input_text = f"grammar: {text_data}"
            result = cls.happy_tt.generate_text(input_text, args=cls.args)

            return JSONResponse(content={
                "original_text": text_data,
                "corrected_text": result.text
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

# 인스턴스 생성이 필요 없으므로 이 줄은 제거합니다.
grammar_corrector = GrammarCorrector()

####