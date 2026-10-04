import os
import yt_dlp
from pydub import AudioSegment
from config import logger, Config

def download_youtube_audio(url: str) -> str:
    """
    Download best-available audio for a YouTube URL using yt-dlp.
    
    Args:
        url (str): The YouTube URL to download.
        
    Returns:
        str: The local file path to the downloaded WAV file.
        
    Raises:
        Exception: If the download fails.
    """
    logger.info(f"Downloading audio from YouTube URL: {url}")
    output_template = os.path.join(Config.DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            filename = os.path.splitext(filename)[0] + ".wav"
        logger.info(f"Successfully downloaded to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download YouTube audio: {e}")
        raise

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to WAV format using pydub.
    
    Args:
        input_path (str): Path to the input audio/video file.
        
    Returns:
        str: Path to the converted WAV file.
        
    Raises:
        Exception: If the conversion fails.
    """
    logger.info(f"Converting file to WAV format: {input_path}")
    try:
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz for Whisper
        audio.export(output_path, format="wav")
        logger.info(f"Successfully converted to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert file to WAV: {e}")
        raise

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """
    Split an audio file into smaller chunks.
    
    Args:
        wav_path (str): Path to the WAV file.
        chunk_minutes (int): Length of each chunk in minutes.
        
    Returns:
        list[str]: A list of file paths to the generated chunks.
    """
    logger.info(f"Chunking audio {wav_path} into {chunk_minutes}-minute segments")
    try:
        audio = AudioSegment.from_wav(wav_path)
        chunk_ms = chunk_minutes * 60 * 1000
        chunks = []

        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start: start + chunk_ms]
            chunk_path = f"{wav_path}_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks
    except Exception as e:
        logger.error(f"Failed to chunk audio: {e}")
        raise

def cleanup_files(file_paths: list[str]) -> None:
    """
    Delete temporary files to free up disk space.
    
    Args:
        file_paths (list[str]): List of file paths to delete.
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Cleaned up file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")

def process_input(source: str) -> dict:
    """
    Main entry point for processing audio input.
    Downloads/converts the source and chunks it.
    
    Args:
        source (str): YouTube URL or local file path.
        
    Returns:
        dict: A dictionary containing the list of chunks and the original wav file 
              for later cleanup.
    """
    logger.info(f"Processing input source: {source}")
    try:
        if source.startswith("http://") or source.startswith("https://"):
            logger.info("Detected URL, attempting to download...")
            wav_path = download_youtube_audio(source)
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Local file not found: {source}")
            logger.info("Detected local file, attempting conversion...")
            wav_path = convert_to_wav(source)

        chunks = chunk_audio(wav_path)
        return {
            "chunks": chunks,
            "original_wav": wav_path
        }
    except Exception as e:
        logger.error(f"Error during audio processing: {e}")
        raise