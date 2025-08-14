import logging
import google.generativeai as genai
from openai import OpenAI
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, config, collection):
        self.config = config
        self.collection = collection

    def get_embedding(self, text, provider):
        """Generates vector embedding for a given text chunk."""
        try:
            if provider == "GEMINI":
                genai.configure(api_key=self.config['GOOGLE_API_KEY'])
                response = genai.embed_content(model="models/embedding-001",
                                               content=text,
                                               task_type="retrieval_document")
                return response['embedding']
            elif provider == "OPENAI":
                client = OpenAI(api_key=self.config['OPENAI_API_KEY'])
                response = client.embeddings.create(input=text, model="text-embedding-3-small")
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def answer_question(self, query, provider):
        """Finds relevant chunks and generates an answer using an LLM."""
        query_embedding = self.get_embedding(query, provider)
        if not query_embedding:
            return "Sorry, I couldn't process your question."

        results = self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 5
                }
            }
        ])

        context = ""
        for doc in results:
            context += doc['chunk_text'] + "\n\n"

        if not context:
            return "I couldn't find any relevant information in the documents."

        prompt = f"""
        Based on the following context from the documents, please answer the question.
        If the context does not contain the answer, say that you couldn't find the information.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        try:
            if provider == "GEMINI":
                genai.configure(api_key=self.config['GOOGLE_API_KEY'])
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                return response.text
            elif provider == "OPENAI":
                client = OpenAI(api_key=self.config['OPENAI_API_KEY'])
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"

class RAGPipeline:
    def __init__(self, config, mongo_client, db, collection, pdf_processor, llm_service):
        self.config = config
        self.mongo_client = mongo_client
        self.db = db
        self.collection = collection
        self.pdf_processor = pdf_processor
        self.llm_service = llm_service

    def process_and_store_chunks(self, file_path, provider):
        """Main function to process a PDF and store its chunks in MongoDB."""
        logger.info(f"Processing {file_path}...")
        document_text = self.pdf_processor.load_and_read_pdf(file_path)
        if not document_text:
            logger.error(f"Failed to read or extract text from {file_path}. Skipping.")
            return

        chunks = self.pdf_processor.chunk_text_by_paragraph(document_text)
        for i, chunk_text in enumerate(chunks):
            embedding = self.llm_service.get_embedding(chunk_text, provider)
            if embedding:
                document = {
                    "file_path": file_path,
                    "chunk_index": i,
                    "chunk_text": chunk_text,
                    "embedding": embedding
                }
                self.collection.insert_one(document)
                logger.info(f"  Stored chunk {i+1}/{len(chunks)}")
            else:
                logger.warning(f"  Skipped chunk {i+1}/{len(chunks)} due to embedding error.")
        logger.info(f"Finished processing {file_path}.")
