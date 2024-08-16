import io
from gtts import gTTS
from fastapi.responses import StreamingResponse

async def generate_speech(text: str, lang: str = 'ko') -> StreamingResponse:
    try:
        # gTTS를 사용하여 음성 생성
        tts = gTTS(text=text, lang=lang)
        
        # 음성 데이터를 바이트 스트림으로 저장
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)

        # StreamingResponse 생성 및 반환
        return StreamingResponse(audio_stream, media_type="audio/mp3")
    
    except Exception as e:
        # 에러 로깅 또는 처리
        print(f"Error in text-to-speech generation: {str(e)}")
        raise