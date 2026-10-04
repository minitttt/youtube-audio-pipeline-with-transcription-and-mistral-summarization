import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging() -> logging.Logger:
    """Sets up and returns a centralized logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("video_agent")

logger = setup_logging()

class Config:
    """Centralized configuration for the Video Agent."""
    # Models
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    
    # Paths
    DOWNLOAD_DIR = os.path.join(os.getcwd(), 'downloads')
    CHROMA_DB_DIR = os.path.join(os.getcwd(), 'vector_db')
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # If used in future
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure critical directories exist
os.makedirs(Config.DOWNLOAD_DIR, exist_ok=True)
os.makedirs(Config.CHROMA_DB_DIR, exist_ok=True)
