import os
import sys
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def get_config():
    load_dotenv() # Ensure .env is loaded
    config = {
        'MONGO_URI': os.getenv('MONGO_URI'),
        'DB_NAME': os.getenv('DB_NAME', 'rag_tutorial'),
        'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'pdf_chunks'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'LLM_PROVIDER': os.getenv('LLM_PROVIDER', 'GEMINI').upper(),
    }
    # Validate required config
    missing = [k for k, v in config.items() if v is None and k in ['MONGO_URI']]
    if missing:
        logger.error(f"Missing required config: {missing}")
        sys.exit(1)
    return config
