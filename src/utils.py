import logging
import sys
import google.generativeai as genai
from pymongo import MongoClient

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def test_mongodb_connection(mongo_client):
    try:
        mongo_client.admin.command('ping')
        logger.info("MongoDB connection successful!")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")

def test_gemini_api_key(api_key):
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set in environment.")
        return
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        logger.info("Gemini API key is valid! Models available:")
        for model in models:
            logger.info(f"  - {model.name}")
    except Exception as e:
        logger.error(f"Gemini API key test failed: {e}")
