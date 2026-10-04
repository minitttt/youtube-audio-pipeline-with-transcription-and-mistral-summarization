import whisper
from config import Config, logger

model = None

def load_model() -> whisper.Whisper:
    """
    Loads and caches the Whisper model in memory.
    
    Returns:
        whisper.Whisper: The loaded Whisper model.
    """
    global model
    if model is None:
        logger.info(f"Loading Whisper model: {Config.WHISPER_MODEL}...")
        try:
            model = whisper.load_model(Config.WHISPER_MODEL)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    return model

def transcribe_audio(chunk_path: str, translate: bool = False) -> str:
    """
    Transcribe a single audio chunk.
    
    Args:
        chunk_path (str): Path to the audio chunk.
        translate (bool): Whether to translate to English.
        
    Returns:
        str: The transcribed text.
    """
    loaded_model = load_model()
    task = "translate" if translate else "transcribe"
    logger.debug(f"Transcribing {chunk_path} with task '{task}'")
    try:
        result = loaded_model.transcribe(chunk_path, task=task)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing {chunk_path}: {e}")
        raise

def transcribe_all(chunks: list[str], translate: bool = False) -> str:
    """
    Transcribe multiple audio chunks and combine the text.
    
    Args:
        chunks (list[str]): List of paths to audio chunks.
        translate (bool): Whether to translate to English.
        
    Returns:
        str: The combined transcription text.
    """
    full_transcription = ""
    total = len(chunks)
    logger.info(f"Starting transcription of {total} chunks.")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Transcribing chunk {i + 1}/{total}...")
        try:
            transcription = transcribe_audio(chunk, translate)
            full_transcription += transcription + " "
        except Exception as e:
            logger.error(f"Skipping chunk {i + 1} due to error: {e}")
            
    logger.info("Transcription completed.")
    return full_transcription.strip()