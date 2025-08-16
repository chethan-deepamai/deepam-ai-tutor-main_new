
"""
RAG (Retrieval-Augmented Generation) System for DeepAM AI Tutor
This module handles document processing, vector storage, and AI-powered responses.
"""

import os
import json
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Document processing
import PyPDF2
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google import genai

# Vector storage and embeddings
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# LangChain for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF processing using Google Document AI and fallback PyPDF2"""
    
    def __init__(self, project_id: str, location: str, processor_id: str):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.client = None
        
        try:
            if project_id and location and processor_id:
                self.client = documentai.DocumentProcessorServiceClient()
                self.processor_name = self.client.processor_path(
                    project_id, location, processor_id
                )
                try:
                    processor_info = self.client.get_processor(name=self.processor_name)
                    logger.info(f"Document AI processor type: {processor_info.type_}")
                    logger.info(f"Document AI processor display name: {processor_info.display_name}")
                    if "OCR" not in processor_info.type_.upper() and "FORM" not in processor_info.type_.upper():
                        logger.warning(f"Processor type '{processor_info.type_}' may not be optimal for text extraction.")
                    logger.info("Document AI client initialized successfully")
                except Exception as info_error:
                    logger.warning(f"Could not get processor info: {info_error}, but proceeding")
        except Exception as e:
            logger.warning(f"Document AI initialization failed: {e}. Using PyPDF2 fallback.")
    
    def process_pdf_document_ai(self, file_content: bytes, mime_type: str = "application/pdf") -> str:
        """Process PDF using Google Document AI with multi-language support"""
        if not self.client:
            raise Exception("Document AI client not initialized")
        
        try:
            raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
            request = documentai.ProcessRequest(
                name=self.processor_name, 
                raw_document=raw_document,
                process_options=documentai.ProcessOptions(
                    ocr_config=documentai.OcrConfig(
                        enable_native_pdf_parsing=True,
                        enable_image_quality_scores=True,
                        enable_symbol=True,
                        premium_features=documentai.OcrConfig.PremiumFeatures(
                            enable_selection_mark_detection=True,
                            compute_style_info=True,
                            enable_math_ocr=False
                        )
                    )
                )
            )
            result = self.client.process_document(request=request)
            document = result.document
            text = document.text
            if text:
                import unicodedata
                text = unicodedata.normalize('NFC', text)
                cleaned_text = "".join(char for char in text if char.isprintable() or char.isspace() or ord(char) > 127)
                text = cleaned_text
                logger.info(f"Document AI extracted {len(text)} characters")
                sample_text = text[:200].replace('\n', '\\n')
                logger.info(f"Sample extracted text: {sample_text}")
            return text
        except Exception as e:
            if "entity_types" in str(e).lower():
                logger.warning("Retrying Document AI with basic OCR request")
                return self.process_pdf_document_ai_basic(file_content, mime_type)
            else:
                raise e
    
    def process_pdf_document_ai_basic(self, file_content: bytes, mime_type: str = "application/pdf") -> str:
        """Basic Document AI processing without advanced options"""
        try:
            raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
            request = documentai.ProcessRequest(name=self.processor_name, raw_document=raw_document)
            result = self.client.process_document(request=request)
            document = result.document
            text = document.text
            if text:
                import unicodedata
                text = unicodedata.normalize('NFC', text)
                logger.info(f"Document AI basic extraction: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Document AI basic processing failed: {e}")
            raise e
    
    def process_pdf_pypdf2(self, file_content: bytes) -> str:
        """Fallback PDF processing using PyPDF2"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            text = ""
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            import unicodedata
                            page_text = unicodedata.normalize('NFC', page_text)
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue
            os.unlink(temp_file_path)
            if not text.strip():
                logger.warning("No text extracted with PyPDF2, trying alternative extraction")
                text = self.process_pdf_alternative(file_content)
            return text
        except Exception as e:
            logger.error(f"PyPDF2 processing failed: {e}")
            raise
    
    def process_pdf_alternative(self, file_content: bytes) -> str:
        """Alternative PDF processing for complex scripts"""
        try:
            import io
            from PyPDF2 import PdfReader
            text = ""
            pdf_stream = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_stream)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        import unicodedata
                        page_text = unicodedata.normalize('NFC', page_text)
                        page_text = ''.join(char for char in page_text if ord(char) > 31 or char in '\t\n\r')
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Alternative extraction failed for page {page_num}: {e}")
                    continue
            return text
        except Exception as e:
            logger.error(f"Alternative PDF processing failed: {e}")
            return ""
    
    def process_document(self, file_content: bytes) -> str:
        """Process document with Document AI fallback to PyPDF2"""
        extracted_text = ""
        if self.client:
            try:
                logger.info("Processing document with Document AI")
                extracted_text = self.process_pdf_document_ai(file_content)
                if extracted_text and len(extracted_text.strip()) > 50:
                    logger.info(f"Document AI extracted {len(extracted_text)} characters")
                    return extracted_text
                else:
                    logger.warning("Document AI extracted insufficient text, trying fallback")
            except Exception as e:
                logger.warning(f"Document AI failed: {e}. Using PyPDF2 fallback")
        logger.info("Processing document with PyPDF2")
        try:
            extracted_text = self.process_pdf_pypdf2(file_content)
            if extracted_text and len(extracted_text.strip()) > 10:
                logger.info(f"PyPDF2 extracted {len(extracted_text)} characters")
                return extracted_text
            else:
                logger.warning("PyPDF2 extracted insufficient text")
        except Exception as e:
            logger.error(f"PyPDF2 failed: {e}")
        if not extracted_text or len(extracted_text.strip()) < 10:
            logger.error("Both Document AI and PyPDF2 failed to extract meaningful text")
            raise Exception("Failed to extract text from document")
        return extracted_text

class VectorStore:
    """Handles vector storage and retrieval using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """Create or get a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
        except:
            collection = self.client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
        return collection
    
    def normalize_language(self, language: str) -> str:
        """Normalize language codes"""
        language_map = {
            'english': 'english', 'en': 'english', 'hindi': 'hindi', 'hi': 'hindi',
            'tamil': 'tamil', 'ta': 'tamil', 'telugu': 'telugu', 'te': 'telugu',
            'kannada': 'kannada', 'kn': 'kannada', 'malayalam': 'malayalam', 'ml': 'malayalam',
            'bengali': 'bengali', 'bn': 'bengali', 'gujarati': 'gujarati', 'gu': 'gujarati',
            'marathi': 'marathi', 'mr': 'marathi', 'punjabi': 'punjabi', 'pa': 'punjabi'
        }
        return language_map.get(language.lower(), language.lower())
    
    def generate_collection_name(self, board: str, subject: str, class_: str, language: str) -> str:
        """Generate a standardized collection name"""
        normalized_language = self.normalize_language(language)
        name = f"{board}_{subject}_class{class_}_{normalized_language}".lower()
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return name[:50]
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the vector store"""
        chunks = self.text_splitter.split_text(text)
        collection_name = self.generate_collection_name(
            metadata['board'], metadata['subject'], metadata['class'], metadata['language']
        )
        collection = self.create_collection(collection_name)
        doc_id = str(uuid.uuid4())
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedding_model.encode(chunk).tolist()
            chunk_metadata = {
                "board": metadata.get('board', ''),
                "subject": metadata.get('subject', ''),
                "class": metadata.get('class', ''),
                "language": metadata.get('language', ''),
                "filename": metadata.get('filename', ''),
                "chunk_id": chunk_id,
                "chunk_index": i,
                "document_id": doc_id
            }
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id]
            )
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks to collection {collection_name}")
        return doc_id
    
    def search_similar(self, query: str, board: str, subject: str, class_: str, 
                      language: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content"""
        collection_name = self.generate_collection_name(board, subject, class_, language)
        try:
            collection = self.client.get_collection(collection_name)
        except:
            logger.warning(f"Collection {collection_name} not found")
            return []
        query_embedding = self.embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        similar_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                similar_docs.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity": max(0, 1 - results['distances'][0][i])
                })
        return similar_docs

class AIResponseGenerator:
    """Generates AI responses using Google's Gemini model (AI Studio key)."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        try:
            api_key = os.getenv("GEMINI_API_KEY")  # set this in your env
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")

            # New SDK pattern: create a Client with the API key
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.0-flash"
            logger.info("AI Response Generator initialized (AI Studio key).")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.client = None
            self.model_name = None

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client or not self.model_name:
            return {"response": "AI not configured. Check GEMINI_API_KEY.", "sources": [], "confidence": "error"}

        if not context_docs:
            msg = (f"I couldn't find specific information about '{query}' in the uploaded materials "
                   f"for {metadata.get('subject','the subject')}. Try rephrasing your question or upload the relevant textbook.")
            return {"response": msg, "sources": [], "confidence": "low"}

        combined_context = "\n\n".join([doc["text"] for doc in context_docs])
        prompt = f"""
You are an expert AI tutor for a 10th-grade student. Your name is DeepAM.

Student:
Board: {metadata.get('board','CBSE/NCERT')}
Subject: {metadata.get('subject','Science')}
Language: {metadata.get('language','English')}

CONTEXT:
---
{combined_context}
---

QUESTION: "{query}"

Give a simple, step-by-step explanation in {metadata.get('language','English')}.
If context is insufficient, say so and suggest what to look for.
"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            # Build sources list from doc metadata
            sources = []
            for d in context_docs:
                md = d.get("metadata", {})
                filename = md.get("filename", "Textbook")
                label = f"{metadata.get('board','CBSE/NCERT')} {metadata.get('subject','Science')} - {filename}"
                if label not in sources:
                    sources.append(label)

            return {
                "response": resp.text,
                "sources": sources,
                "context_used": len(context_docs),
                "confidence": "high"
            }
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return {"response": f"Error generating response for '{query}'.", "sources": [], "confidence": "error", "error_details": str(e)}

class RAGSystem:
    """Main RAG system that orchestrates document processing, vector storage, and AI responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.document_processor = DocumentProcessor(
            project_id=config.get('project_id'),
            location=config.get('doc_ai_location', 'us'),
            processor_id=config.get('doc_ai_processor_id')
        )
        self.vector_store = VectorStore(
            persist_directory=config.get('vector_db_path', './chroma_db')
        )
        self.ai_generator = AIResponseGenerator(
            project_id=config.get('project_id'),
            location=config.get('location', 'us-central1')
        )
        logger.info("RAG System initialized successfully")
    
    def process_and_store_document(self, file_content: bytes, metadata: Dict[str, Any]) -> str:
        """Process a document and store it in the vector database"""
        try:
            logger.info(f"Processing document: {metadata.get('filename', 'unknown')}")
            text = self.document_processor.process_document(file_content)
            if not text or len(text.strip()) < 50:
                raise Exception("Document appears to be empty or contains insufficient text")
            doc_id = self.vector_store.add_document(text, metadata)
            logger.info(f"Document processed and stored successfully: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to process and store document: {e}")
            raise
    
    def generate_answer(self, query: str, board: str, subject: str, class_: str, 
                       language: str, user_id: str) -> Dict[str, Any]:
        """Generate an AI-powered answer using RAG"""
        try:
            logger.info(f"Searching for context: {query}")
            similar_docs = self.vector_store.search_similar(
                query=query,
                board=board,
                subject=subject,
                class_=class_,
                language=language,
                top_k=5
            )
            metadata = {
                "board": board,
                "subject": subject,
                "class": class_,
                "language": language,
                "user_id": user_id
            }
            logger.info(f"Generating AI response with {len(similar_docs)} context documents")
            result = self.ai_generator.generate_response(query, similar_docs, metadata)
            result["query_id"] = str(uuid.uuid4())
            result["metadata"] = metadata
            return result
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "response": f"I apologize, but I encountered an error while processing your question about '{query}'. Please try again or contact support.",
                "sources": [],
                "query_id": str(uuid.uuid4()),
                "metadata": {
                    "board": board,
                    "subject": subject,
                    "class": class_,
                    "language": language,
                    "user_id": user_id
                },
                "error_details": str(e),
                "context_used": 0,
                "confidence": "low"
            }

def create_rag_system() -> RAGSystem:
    """Factory function to create a RAG system with configuration"""
    config = {
        'project_id': os.getenv('PROJECT_ID', 'deepam-ai-tutor'),
        'location': os.getenv('REGION', 'asia-south1'),
        'doc_ai_location': os.getenv('DOC_AI_LOCATION', 'us'),
        'doc_ai_processor_id': os.getenv('DOC_AI_PROCESSOR_ID'),
        'vector_db_path': os.getenv('VECTOR_DB_PATH', './chroma_db')
    }
    return RAGSystem(config)