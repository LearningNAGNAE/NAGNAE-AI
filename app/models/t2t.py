import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class T2T:
    tokenizer = None
    model = None

    @classmethod
    def load_model(cls):
        if cls.tokenizer is None or cls.model is None:
            print("Loading model and tokenizer...")
            cls.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            cls.model = T5ForConditionalGeneration.from_pretrained("t5-base")
            print("Model and tokenizer loaded.")

    @classmethod
    async def generate_code(cls, text_data):
        cls.load_model()  # 필요할 때만 모델을 로드합니다.

        questions = [line.strip() for line in text_data.split("\n")]

        results = []
        for question in questions:
            inputs = cls.tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            summary_ids = cls.model.generate(**inputs)
            summary = cls.tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
            results.append(summary)

        python_code = "".join(results)
        return {"result": python_code}

# 클래스 메서드를 직접 사용할 수 있으므로, 인스턴스 생성이 필요 없습니다.
t2t = T2T()  # 이 줄은 제거해도 됩니다.