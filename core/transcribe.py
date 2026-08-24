import whisper
import os
from utlis.audio_processor import process_input


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


model = None

def load_model():
    global model
    if model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        model = whisper.load_model(WHISPER_MODEL)
    return model

def transcribe_audio(chunk_path: str, translate:bool=False)-> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    return result["text"]
    

def transcribe_all(chunks: list, translate=False) -> str:
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk}")
        transcription = transcribe_audio(chunk, translate)
        full_transcription += transcription + " "
    return full_transcription.strip()   