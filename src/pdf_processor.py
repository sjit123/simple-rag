import logging
import pypdf

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        pass

    def load_and_read_pdf(self, file_path):
        """Extracts text from a PDF file."""
        try:
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return None

    def chunk_text_by_paragraph(self, text, min_chunk_size=50, max_chunk_size=8000):
        """Splits text into chunks, first by paragraphs, then by size."""
        paragraphs = text.split('\n\n')
        chunks = []
        for paragraph in paragraphs:
            p_stripped = paragraph.strip()
            if len(p_stripped) < min_chunk_size:
                continue
            if len(p_stripped) <= max_chunk_size:
                chunks.append(p_stripped)
            else:
                # If a paragraph is too long, split it into smaller chunks
                for i in range(0, len(p_stripped), max_chunk_size):
                    chunks.append(p_stripped[i:i + max_chunk_size])
        return chunks
