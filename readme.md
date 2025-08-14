
# RAG (Retrieval-Augmented Generation) Pipeline Demo

A modular, production-ready RAG pipeline for PDF document processing, semantic search, and LLM-powered Q&A using MongoDB Atlas, Gemini, and OpenAI.

## Features

- **PDF Processing**: Extracts and chunks text from PDF files
- **Multiple LLM Support**: Gemini and OpenAI integration
- **Vector Search**: Semantic search via MongoDB Atlas
- **Scalable Storage**: Chunks and embeddings stored in MongoDB
- **Interactive Q&A**: Context-aware answers from indexed documents
- **CLI Interface**: Simple commands for indexing and querying
- **Robust Logging**: Detailed logs for monitoring and debugging

## Project Structure

- `src/config.py`: Loads and validates environment variables
- `src/utils.py`: Logging setup and test utilities
- `src/pdf_processor.py`: PDF extraction and chunking logic
- `src/llm_service.py`: Embedding generation, Q&A, and pipeline orchestration
- `src/rag_pipleline_demo.py`: Main CLI entry point

## Prerequisites

- Python 3.8+
- MongoDB Atlas cluster with vector search
- Gemini or OpenAI API keys
- PDF documents to process

## Installation

```bash
git clone <your-repository-url>
cd gen-ai-stuff
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
MONGO_URI=your_mongodb_atlas_connection_string
DB_NAME=rag_tutorial
COLLECTION_NAME=pdf_chunks
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
LLM_PROVIDER=GEMINI  # or OPENAI
```

Set up MongoDB Vector Search Index (Atlas):

```javascript
{
   "fields": [
      {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 768,  // Gemini: 768, OpenAI: 1536
         "similarity": "cosine"
      }
   ]
}
```

## Usage

### Test Configuration
Verify MongoDB and API keys:
```bash
python src/rag_pipleline_demo.py --test
```

### Index a PDF
```bash
python src/rag_pipleline_demo.py --index path/to/document.pdf
```

### Start Q&A Session
```bash
python src/rag_pipleline_demo.py --qa
```

If no arguments are provided, the CLI will display usage instructions and available options.

## Architecture Overview

1. **PDF Processing**: Extracts and chunks text for embedding
2. **Vector Storage**: Stores embeddings in MongoDB Atlas for semantic search
3. **Q&A**: Retrieves relevant chunks and generates answers using LLMs

## Error Handling & Logging

- Handles invalid API keys, MongoDB issues, PDF errors, and large chunk management
- Logs document processing, embedding generation, DB operations, API calls, and errors

## Troubleshooting

- **MongoDB**: Check connection string, network, and IP whitelist
- **API Keys**: Validate keys and environment variables
- **PDFs**: Ensure files are accessible and not corrupted
- **Vector Search**: Confirm index and dimension settings

