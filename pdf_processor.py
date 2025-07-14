import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# PDF Processing
import PyPDF2
from pdfplumber import PDF

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    """Data class for PDF document information"""
    filename: str
    content: str
    metadata: Dict
    page_count: int


class PDFVectorStore:
    """
    Handles PDF processing and vector store creation for therapy documents.
    Converts PDF content into searchable embeddings for RAG-based AI responses.
    """
    
    def __init__(self, folder_path: str = "./pdf/", openai_api_key: str = None):
        """
        Initialize PDF Vector Store
        
        Args:
            folder_path: Path to folder containing PDF files
            openai_api_key: OpenAI API key for embeddings
        """
        self.folder_path = folder_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.documents: List[PDFDocument] = []
        self.vector_store: Optional[FAISS] = None
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Ensure PDF directory exists
        os.makedirs(folder_path, exist_ok=True)
        
    def load_pdf_files(self) -> List[PDFDocument]:
        """
        Load all PDF files from the specified directory
        
        Returns:
            List of PDFDocument objects
        """
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.folder_path}")
            return []
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.folder_path, pdf_file)
            try:
                # Try pdfplumber first (better for complex PDFs)
                content = self._extract_with_pdfplumber(file_path)
                if not content:
                    # Fallback to PyPDF2
                    content = self._extract_with_pypdf2(file_path)
                
                if content:
                    # Get page count
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        page_count = len(pdf_reader.pages)
                    
                    doc = PDFDocument(
                        filename=pdf_file,
                        content=content,
                        metadata={
                            'source': pdf_file,
                            'therapy_type': self._infer_therapy_type(pdf_file, content)
                        },
                        page_count=page_count
                    )
                    self.documents.append(doc)
                    logger.info(f"Successfully loaded: {pdf_file} ({page_count} pages)")
                else:
                    logger.error(f"Could not extract content from: {pdf_file}")
                    
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
                
        return self.documents
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            with PDF.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 as fallback"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def _infer_therapy_type(self, filename: str, content: str) -> str:
        """
        Infer therapy type from filename or content
        
        Args:
            filename: Name of the PDF file
            content: Text content of the PDF
            
        Returns:
            Inferred therapy type
        """
        filename_lower = filename.lower()
        content_lower = content.lower()[:1000]  # Check first 1000 chars
        
        therapy_keywords = {
            'cbt': ['cognitive', 'behavioral', 'cbt', 'thought'],
            'dbt': ['dialectical', 'dbt', 'mindfulness', 'distress tolerance'],
            'act': ['acceptance', 'commitment', 'act', 'values'],
            'grief': ['grief', 'loss', 'bereavement', 'mourning'],
            'anxiety': ['anxiety', 'worry', 'panic', 'fear'],
            'parenting': ['parenting', 'child', 'family', 'parent'],
            'depression': ['depression', 'mood', 'sadness'],
            'trauma': ['trauma', 'ptsd', 'stress', 'traumatic']
        }
        
        for therapy_type, keywords in therapy_keywords.items():
            if any(keyword in filename_lower or keyword in content_lower 
                   for keyword in keywords):
                return therapy_type
                
        return 'general'
    
    def build_vector_store(self) -> FAISS:
        """
        Build FAISS vector store from loaded documents
        
        Returns:
            FAISS vector store instance
        """
        if not self.documents:
            self.load_pdf_files()
            
        if not self.documents:
            raise ValueError("No documents loaded. Please add PDF files to the folder.")
        
        # Convert PDFDocuments to LangChain Documents
        langchain_docs = []
        
        for pdf_doc in self.documents:
            # Split the document into chunks
            chunks = self.text_splitter.split_text(pdf_doc.content)
            
            # Create LangChain Document for each chunk
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **pdf_doc.metadata,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                )
                langchain_docs.append(doc)
        
        logger.info(f"Created {len(langchain_docs)} document chunks from {len(self.documents)} PDFs")
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings
        )
        
        # Save the vector store for future use
        self.save_vector_store()
        
        return self.vector_store
    
    def save_vector_store(self, path: str = "./vector_store/"):
        """Save vector store to disk"""
        if self.vector_store:
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str = "./vector_store/"):
        """Load vector store from disk"""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
            logger.info(f"Vector store loaded from {path}")
            return True
        return False
    
    def get_similar_docs(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar documents based on query
        
        Args:
            query: User query text
            k: Number of similar documents to retrieve
            
        Returns:
            List of similar documents with metadata
        """
        if not self.vector_store:
            # Try to load from disk first
            if not self.load_vector_store():
                # Build if not available
                self.build_vector_store()
        
        # Perform similarity search
        similar_docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in similar_docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score),
                'source': doc.metadata.get('source', 'Unknown'),
                'therapy_type': doc.metadata.get('therapy_type', 'general')
            })
        
        return results
    
    def get_context_for_prompt(self, query: str, k: int = 3) -> str:
        """
        Get formatted context string for prompt generation
        
        Args:
            query: User query
            k: Number of documents to include
            
        Returns:
            Formatted context string
        """
        similar_docs = self.get_similar_docs(query, k)
        
        if not similar_docs:
            return "No relevant therapy context found."
        
        context_parts = []
        for i, doc in enumerate(similar_docs, 1):
            context_parts.append(
                f"Reference {i} (from {doc['source']} - {doc['therapy_type']} therapy):\n"
                f"{doc['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        stats = {
            'total_pdfs': len(self.documents),
            'total_pages': sum(doc.page_count for doc in self.documents),
            'therapy_types': {}
        }
        
        # Count therapy types
        for doc in self.documents:
            therapy_type = doc.metadata.get('therapy_type', 'unknown')
            stats['therapy_types'][therapy_type] = stats['therapy_types'].get(therapy_type, 0) + 1
        
        if self.vector_store:
            stats['total_chunks'] = self.vector_store.index.ntotal
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Initialize with your OpenAI API key
    pdf_store = PDFVectorStore(
        folder_path="./pdf/",
        openai_api_key="your-openai-api-key"
    )
    
    # Load and process PDFs
    documents = pdf_store.load_pdf_files()
    print(f"Loaded {len(documents)} PDF files")
    
    # Build vector store
    vector_store = pdf_store.build_vector_store()
    print("Vector store built successfully")
    
    # Test similarity search
    test_query = "How to deal with anxiety?"
    results = pdf_store.get_similar_docs(test_query)
    print(f"\nResults for query: '{test_query}'")
    for result in results:
        print(f"- Source: {result['source']}, Score: {result['similarity_score']:.3f}")