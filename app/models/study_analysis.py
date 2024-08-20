import os
import tempfile
import traceback
import numpy as np
import torch
import librosa
import ffmpeg
import io
from gtts import gTTS
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
from io import BytesIO

load_dotenv()

# Constants and model setup
TARGET_SR = 16000
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ffmpeg_path = os.path.join(project_root, "ffmpeg-2024-08-15-git-1f801dfdb5-full_build", "bin")
ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")

# LangChain setup
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0, 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

async def text_to_speech(text: str, lang: str = 'ko') -> StreamingResponse:
    try:
        tts = gTTS(text=text, lang=lang)
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        return StreamingResponse(audio_stream, media_type="audio/mp3")
    except Exception as e:
        print(f"Error in text-to-speech generation: {str(e)}")
        raise

async def speech_to_text(file_content: bytes):
    temp_webm = None
    temp_wav = None
    try:
        temp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        
        temp_webm.write(file_content)
        temp_webm.close()
        temp_wav.close()
        
        temp_webm_path, temp_wav_path = temp_webm.name, temp_wav.name

        convert_webm_to_wav(temp_webm_path, temp_wav_path)
        audio = load_and_normalize_audio(temp_wav_path)
        transcription = transcribe_audio(audio)

        return transcription, audio

    except Exception as e:
        print(f"Error in speech-to-text conversion: {str(e)}")
        print(traceback.format_exc())
        raise
    finally:
        if temp_webm:
            try:
                os.unlink(temp_webm.name)
            except PermissionError:
                print(f"Unable to delete temporary file: {temp_webm.name}")
        if temp_wav:
            try:
                os.unlink(temp_wav.name)
            except PermissionError:
                print(f"Unable to delete temporary file: {temp_wav.name}")
        torch.cuda.empty_cache()

def convert_webm_to_wav(input_path: str, output_path: str):
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar=TARGET_SR)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, cmd=ffmpeg_exe)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise Exception("FFmpeg processing failed")

def load_and_normalize_audio(file_path: str) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=TARGET_SR)
    return librosa.util.normalize(audio.astype(np.float32))

def transcribe_audio(audio: np.ndarray) -> str:
    max_length = 30 * TARGET_SR
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    input_features = whisper_processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features 
    
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features,
            task="transcribe",
            language="ko",
            max_length=448
        )
    
    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def extract_audio_features(audio, sr):
    # 기본 특성
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    
    # 축소된 MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=7)
    
    # 스펙트럴 롤오프만 유지
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    return {
        "pitch": float(pitch),
        "spectral_centroid": float(np.mean(cent)),
        "rms_energy": float(np.mean(rms)),
        "zero_crossing_rate": float(np.mean(zcr)),
        "tempo": float(tempo)
    }

def create_audio_process_prompts(features, transcription):
    system_template = """
    # AI Assistant for Foreign Workers and Students in Korea: Detailed Korean accent analysis

    ## Role and Responsibility
    You are an expert in Korean accent analysis. Your task is to provide a personalized analysis and feedback on the speaker's Korean pronunciation and intonation.
    
    1. Based on the given audio characteristics and transcribed text, analyze the intonation and overall speech patterns of the Korean speech.
    2. Structure your analysis to directly address the speaker, using "you" and "your" to make it more personal.
    3. For each aspect, describe the current state of the speaker's speech and suggest how it could be improved.
    4. Avoid technical terms and use language that is easy for non-experts to understand.
    5. Use the provided audio features to support your analysis, but do not mention specific numerical values. Instead, describe characteristics in relative terms (e.g., high, low, moderate).
    6. Conclude with specific, actionable advice for overall improvement.
    7. Provide your analysis ONLY in the following format, with no additional comments or explanations:

    Structure your analysis according to the following format:

    1. Emphasis and Intonation Patterns:
    Your emphasis and intonation patterns are [description]. This makes your speech sound [effect]. Try to [suggestion] to enhance your intonation.

    2. Speech Rate and Rhythm:
    Your speaking speed and rhythm are [description]. This impacts your speech by [effect]. You might want to [suggestion] to improve your rhythm.

    3. Emotional and Attitudinal Expression:
    Your emotional expression through intonation is [description]. This conveys [effect] to listeners. To express emotions more effectively, try to [suggestion].

    4. Pronunciation Clarity:
    Your pronunciation clarity is [description]. This affects your speech by [effect]. To improve clarity, focus on [suggestion].

    5. Comparison with Native Speakers:
    Compared to native Korean speakers, your speech [description]. To sound more natural, you could [suggestion].

    6. Overall Assessment and Advice:
    Overall, your Korean pronunciation and intonation [summary assessment]. Here are key points to focus on for improvement:
    - [Key point 1]
    - [Key point 2]
    - [Key point 3]

    Remember, consistent practice is key to improving your Korean pronunciation and intonation. Keep up the good work!
    """

    human_template = f"""
    The following are characteristics extracted from a Korean speech file:

    1. Average Pitch: {features['pitch']:.2f} Hz
    2. Spectral Centroid: {features['spectral_centroid']:.2f}
    3. RMS Energy: {features['rms_energy']:.2f}
    4. Zero Crossing Rate: {features['zero_crossing_rate']:.2f}
    5. Tempo: {features['tempo']:.2f} BPM

    Transcribed text: "{transcription}"

    Please use these numerical values as a reference for your analysis, but do not mention the specific numbers in your response. Instead, interpret them in relative terms (e.g., low, medium, high) based on your expertise in speech analysis. Your analysis should describe the speech characteristics without revealing the exact measurements provided. Address the speaker directly and provide personalized feedback and suggestions for improvement.
    """
    
    return system_template, human_template

def create_image_process_prompts(image_description):
    system_template = """
    You are an expert in transforming image descriptions into natural Korean sentences.
    Your task is to create a descriptive Korean sentence based on the given image description.
    Follow these guidelines strictly:
    1. The sentence should be in a narrative style, as if you're describing the scene to someone.
    2. The description should be concise while including the main content of the image.
    3. Use natural Korean expressions and avoid direct translations.
    4. Provide only the requested Korean sentence without any additional explanation or comments.
    """

    human_template = f"""
    Please transform the following image description into a descriptive Korean sentence:
    "{image_description}"
    Provide your response as one natural, narrative-style sentence in Korean. 
    Ensure that the sentence ends with a verb or an adjective functioning as a verb.
    Do not include any explanations or additional comments, just provide the Korean sentence.
    """
    
    return system_template, human_template

def describe_image(image_content: bytes):
    try:
        image_file = BytesIO(image_content)
        raw_image = Image.open(image_file).convert('RGB')

        inputs = blip_processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Error occurred while describing the image: {str(e)}")
        raise

async def study_text_analysis(file_content: bytes):
    try:
        transcription, audio = await speech_to_text(file_content)
        features = extract_audio_features(audio, TARGET_SR)
        system_template, human_template = create_audio_process_prompts(features, transcription)
        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=human_template)
        ]
        response = llm(messages)
        analysis = response.content.strip()
        return {"transcription": transcription, "analysis": analysis}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise

async def study_image_analysis(audio_content: bytes, image_content: bytes):
    try:
        image_description = describe_image(image_content)
        print(f"Image description: {image_description}")
        image_system_template, image_human_template = create_image_process_prompts(image_description)
        image_messages = [
            SystemMessage(content=image_system_template),
            HumanMessage(content=image_human_template)
        ]
        image_response = llm(image_messages)
        recommend = image_response.content.strip()

        transcription, audio = await speech_to_text(audio_content)
        features = extract_audio_features(audio, TARGET_SR)
        audio_system_template, audio_human_template = create_audio_process_prompts(features, transcription)
        audio_messages = [
            SystemMessage(content=audio_system_template),
            HumanMessage(content=audio_human_template)
        ]
        audio_response = llm(audio_messages)
        analysis = audio_response.content.strip()

        return {"transcription": transcription, "analysis": analysis, "recommend": recommend}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.format_exc())
        raise