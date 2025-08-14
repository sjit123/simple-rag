import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient

from config import get_config
from utils import setup_logging, test_mongodb_connection, test_gemini_api_key
from pdf_processor import PDFProcessor
from llm_service import LLMService, RAGPipeline


# --- LOGGING SETUP ---
setup_logging()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
config = get_config()

# --- Initialize MongoDB Client ---
mongo_client = MongoClient(config['MONGO_URI'])
db = mongo_client[config['DB_NAME']]
collection = db[config['COLLECTION_NAME']]

# --- Initialize Services ---
pdf_processor = PDFProcessor()
llm_service = LLMService(config, collection)
rag_pipeline = RAGPipeline(config, mongo_client, db, collection, pdf_processor, llm_service)



def main():
    """
    Main entry point for the RAG Pipeline Demo.

    This function provides a command-line interface to:
    - Index a PDF file into MongoDB for retrieval-augmented generation (RAG).
    - Start an interactive Q&A loop using an LLM over indexed documents.
    - Test MongoDB connectivity and Gemini API key configuration.

    Usage:
        python rag_pipleline_demo.py --index <PDF_PATH>
            Index the specified PDF file into MongoDB.

        python rag_pipleline_demo.py --qa
            Start an interactive Q&A session with the indexed documents.

        python rag_pipleline_demo.py --test
            Test MongoDB connection and Gemini API key.

    If no arguments are provided, the help message is displayed.
    """
    parser = argparse.ArgumentParser(description="RAG Pipeline Demo: PDF to MongoDB to LLM Q&A")
    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    parser.add_argument('--index', type=str, help='Path to PDF file to index')
    parser.add_argument('--qa', action='store_true', help='Start Q&A loop')
    parser.add_argument('--test', action='store_true', help='Test MongoDB and API keys')
    args = parser.parse_args()

    if args.test:
        logger.info("Testing MongoDB connection...")
        test_mongodb_connection(mongo_client)
        logger.info("Testing Gemini API key...")
        test_gemini_api_key(config['GOOGLE_API_KEY'])
        return

    if args.index:
        pdf_file_path = args.index
        if collection.count_documents({"file_path": pdf_file_path}) == 0:
            logger.info(f"Indexing {pdf_file_path}...")
            rag_pipeline.process_and_store_chunks(pdf_file_path, config['LLM_PROVIDER'])
            logger.info("Indexing complete.")
        else:
            logger.info(f"'{pdf_file_path}' has already been indexed.")

    if args.qa:
        logger.info("--- Ready to answer questions! ---")
        while True:
            user_query = input("Ask a question about the document (or type 'exit'): ")
            if user_query.lower() == 'exit':
                break
            answer = llm_service.answer_question(user_query, config['LLM_PROVIDER'])
            print("\nAnswer:")
            print(answer)
            print("-" * 20)

if __name__ == "__main__":
    main()
