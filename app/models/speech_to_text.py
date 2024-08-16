import os
import tempfile
import traceback
import numpy as np
import torch
import librosa
import ffmpeg
from transformers import WhisperProcessor, WhisperForConditionalGeneration

TARGET_SR = 16000
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# 프로젝트 루트 디렉토리 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ffmpeg_path = os.path.join(project_root, "ffmpeg-2024-08-15-git-1f801dfdb5-full_build", "bin")
ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")

async def speech_to_text(file_content: bytes):
    temp_webm = None
    temp_wav = None
    try:
        # 임시 webm 파일 생성
        temp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_webm.write(file_content)
        temp_webm.close()
        temp_webm_path = temp_webm.name

        # 임시 wav 파일 생성
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        temp_wav_path = temp_wav.name

        try:
            (
                ffmpeg
                .input(temp_webm_path)
                .output(temp_wav_path, acodec='pcm_s16le', ac=1, ar=TARGET_SR)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, cmd=ffmpeg_exe)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise Exception("FFmpeg processing failed")

        # wav 파일 로드 및 처리
        audio, _ = librosa.load(temp_wav_path, sr=TARGET_SR)
        audio = librosa.util.normalize(audio.astype(np.float32))
        
        # 입력 특성 생성 및 텍스트 생성
        input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features 
        
        with torch.no_grad():
            predicted_ids_ko = model.generate(input_features, task="transcribe", language="ko")
        
        transcription = processor.batch_decode(predicted_ids_ko, skip_special_tokens=True)[0]
        return transcription

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise

    finally:
        # 임시 파일 삭제
        if temp_webm and os.path.exists(temp_webm.name):
            os.unlink(temp_webm.name)
        if temp_wav and os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)
        
        # GPU 메모리 정리 (필요한 경우)
        torch.cuda.empty_cache()