# import os
# import traceback
# from fastapi import FastAPI, File, UploadFile, HTTPException
# import torch
# import librosa
# import numpy as np
# from fastapi.middleware.cors import CORSMiddleware
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
# import ffmpeg
# import tempfile

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# app = FastAPI()

# # CORS 설정
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 디바이스 설정
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # 모델과 프로세서 로드
# model_id = "openai/whisper-large-v3"
# processor = WhisperProcessor.from_pretrained(model_id)
# model = WhisperForConditionalGeneration.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# # 파이프라인 설정
# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# TARGET_SR = 16000

# @app.post("/api/automaticspeechrecognition")
# async def transcribe_audio(file: UploadFile = File(..., max_size=1024*1024*10)):  # 10MB로 제한
#     try:
#         # 임시 파일 생성
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
#             temp_webm.write(await file.read())
#             temp_webm_path = temp_webm.name

#         # wav 파일로 변환
#         temp_wav_path = temp_webm_path.replace(".webm", ".wav")
        
#         (
#             ffmpeg
#             .input(temp_webm_path)
#             .output(temp_wav_path, acodec='pcm_s16le', ac=1, ar=TARGET_SR)
#             .overwrite_output()
#             .run(capture_stdout=True, capture_stderr=True)
#         )

#         # wav 파일 로드
#         audio, _ = librosa.load(temp_wav_path, sr=TARGET_SR)
        
#         # float32로 변환 및 정규화
#         audio = librosa.util.normalize(audio.astype(np.float32))
        
#         # 파이프라인을 사용하여 오디오를 텍스트로 변환
#         inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
#         input_features = inputs.input_features.to(device)
        
#         # 영어 텍스트 생성
#         with torch.no_grad():
#             predicted_ids_en = model.generate(input_features, max_length=448)
        
#         # 토큰 ID를 텍스트로 디코딩
#         transcription = processor.batch_decode(predicted_ids_en, skip_special_tokens=True)[0]

#         # 임시 파일 삭제
#         os.unlink(temp_webm_path)
#         os.unlink(temp_wav_path)

#         return {"transcription": transcription}

#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
