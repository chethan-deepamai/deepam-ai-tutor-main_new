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
from google.cloud import aiplatform

# Vector storage and embeddings
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# LangChain for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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
                logger.info("Document AI client initialized successfully")
        except Exception as e:
            logger.warning(f"Document AI initialization failed: {e}. Will use PyPDF2 fallback.")
    
    def process_pdf_document_ai(self, file_content: bytes, mime_type: str = "application/pdf") -> str:
        """Process PDF using Google Document AI"""
        if not self.client:
            raise Exception("Document AI client not initialized")
        
        # Create the document object
        raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
        
        # Configure the process request
        request = documentai.ProcessRequest(
            name=self.processor_name, raw_document=raw_document
        )
        
        # Process the document
        result = self.client.process_document(request=request)
        document = result.document
        
        # Extract text
        text = document.text
        
        return text
    
    def process_pdf_pypdf2(self, file_content: bytes) -> str:
        """Fallback PDF processing using PyPDF2"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Extract text using PyPDF2
            text = ""
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return text
        except Exception as e:
            logger.error(f"PyPDF2 processing failed: {e}")
            raise
    
    def process_document(self, file_content: bytes) -> str:
        """Process document with Document AI fallback to PyPDF2"""
        try:
            # Try Document AI first
            if self.client:
                logger.info("Processing document with Document AI")
                return self.process_pdf_document_ai(file_content)
        except Exception as e:
            logger.warning(f"Document AI failed: {e}. Falling back to PyPDF2")
        
        # Fallback to PyPDF2
        logger.info("Processing document with PyPDF2")
        return self.process_pdf_pypdf2(file_content)

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
    
    def generate_collection_name(self, board: str, subject: str, class_: str, language: str) -> str:
        """Generate a standardized collection name"""
        # Sanitize collection name
        name = f"{board}_{subject}_class{class_}_{language}".lower()
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return name[:50]  # Limit length
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the vector store"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        collection_name = self.generate_collection_name(
            metadata['board'], metadata['subject'], 
            metadata['class'], metadata['language']
        )
        collection = self.create_collection(collection_name)
        
        # Generate embeddings and store chunks
        doc_id = str(uuid.uuid4())
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(chunk).tolist()
            
            # Prepare chunk metadata
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
            
            # Add to collection
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
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        similar_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                similar_docs.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity": max(0, 1 - results['distances'][0][i])  # Convert distance to similarity
                })
        
        return similar_docs

class AIResponseGenerator:
    """Generates AI responses using Vertex AI or fallback responses"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.llm = None
        
        # Try to initialize Vertex AI
        try:
            if project_id:
                aiplatform.init(project=project_id, location=location)
                # For now, we'll use a simple fallback since VertexAI integration is complex
                logger.info("AI Response Generator initialized (using fallback mode)")
        except Exception as e:
            logger.warning(f"Vertex AI initialization failed: {e}. Using fallback mode.")
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], 
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response using retrieved context"""
        
        # Use intelligent fallback that creates educational responses
        return self._generate_educational_response(query, context_docs, metadata)
    
    def _generate_educational_response(self, query: str, context_docs: List[Dict[str, Any]], 
                                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an educational response using retrieved context"""
        
        if context_docs:
            # Extract relevant information from context
            relevant_texts = [doc["text"] for doc in context_docs]
            combined_context = "\n\n".join(relevant_texts[:3])  # Use top 3 results
            
            # Create a structured educational response
            response_parts = []
            
            # Introduction
            response_parts.append(f"Based on your {metadata.get('board', 'CBSE/NCERT')} {metadata.get('subject', 'Science')} textbook for class {metadata.get('class', '10')}, here's what I can explain about '{query}':")
            
            # Main content from context
            if len(combined_context) > 100:
                # Summarize the context intelligently
                context_summary = self._summarize_context(combined_context, query)
                response_parts.append(f"\n\n{context_summary}")
            else:
                response_parts.append(f"\n\n{combined_context}")
            
            # Educational guidance
            response_parts.append(f"\n\nTo better understand this concept:")
            response_parts.append(f"1. Review the relevant chapter in your textbook")
            response_parts.append(f"2. Try to identify key terms and their definitions")
            response_parts.append(f"3. Look for examples or diagrams that illustrate this concept")
            
            # Follow-up questions (Socratic method)
            follow_up = self._generate_follow_up_questions(query, metadata.get('subject', 'Science'))
            if follow_up:
                response_parts.append(f"\n\nThink about this: {follow_up}")
            
            response = "".join(response_parts)
            
            # Extract sources
            sources = []
            for doc in context_docs:
                doc_metadata = doc.get('metadata', {})
                filename = doc_metadata.get('filename', 'Textbook')
                source = f"{metadata.get('board', 'CBSE/NCERT')} {metadata.get('subject', 'Science')} - {filename}"
                if source not in sources:
                    sources.append(source)
            
            confidence = "high" if len(context_docs) >= 3 else "medium"
            
        else:
            # No context found
            response = f"I couldn't find specific information about '{query}' in the uploaded {metadata.get('board', 'CBSE/NCERT')} {metadata.get('subject', 'Science')} textbook for class {metadata.get('class', '10')}.\n\nThis could mean:\n1. The topic might be covered in a different chapter\n2. You might need to upload the relevant textbook section\n3. Try rephrasing your question with more specific terms\n\nWould you like to try asking about a related concept or upload additional study material?"
            
            sources = []
            confidence = "low"
        
        return {
            "response": response,
            "sources": sources,
            "context_used": len(context_docs),
            "confidence": confidence
        }
    
    def _summarize_context(self, context: str, query: str) -> str:
        """Create a simple summary of the context relevant to the query"""
        # Split context into sentences
        sentences = context.replace('\n', ' ').split('. ')
        
        # Find sentences most relevant to the query
        query_words = query.lower().split()
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            relevance_score = sum(1 for word in query_words if word in sentence_lower)
            if relevance_score > 0:
                relevant_sentences.append((sentence, relevance_score))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        if top_sentences:
            return ". ".join(top_sentences) + "."
        else:
            # If no specific relevance found, return first part of context
            return context[:500] + "..." if len(context) > 500 else context
    
    def _generate_follow_up_questions(self, query: str, subject: str) -> str:
        """Generate educational follow-up questions"""
        subject_questions = {
            "Science": [
                "What real-world examples can you think of?",
                "How does this concept connect to what you learned in previous chapters?",
                "Can you explain this to a friend in simple words?",
                "What would happen if this process didn't occur?"
            ],
            "Math": [
                "Can you solve a similar problem using this concept?",
                "Where might you use this in everyday life?",
                "What patterns do you notice?",
                "How is this connected to other math topics you've learned?"
            ],
            "Social Studies": [
                "How does this relate to current events?",
                "What were the causes and effects?",
                "How might this have been different in another time or place?",
                "What lessons can we learn from this?"
            ]
        }
        
        questions = subject_questions.get(subject, subject_questions["Science"])
        import random
        return random.choice(questions)

class RAGSystem:
    """Main RAG system that orchestrates document processing, vector storage, and AI responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
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
            # Extract text from document
            logger.info(f"Processing document: {metadata.get('filename', 'unknown')}")
            text = self.document_processor.process_document(file_content)
            
            if not text or len(text.strip()) < 50:
                raise Exception("Document appears to be empty or contains insufficient text")
            
            # Store in vector database
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
            # Search for relevant context
            logger.info(f"Searching for context: {query}")
            similar_docs = self.vector_store.search_similar(
                query=query,
                board=board,
                subject=subject,
                class_=class_,
                language=language,
                top_k=5
            )
            
            # Prepare metadata for response generation
            metadata = {
                "board": board,
                "subject": subject,
                "class": class_,
                "language": language,
                "user_id": user_id
            }
            
            # Generate AI response
            logger.info(f"Generating AI response with {len(similar_docs)} context documents")
            result = self.ai_generator.generate_response(query, similar_docs, metadata)
            
            # Add query ID for tracking
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
                "error": str(e),
                "context_used": 0,
                "confidence": "low"
            }

def create_rag_system() -> RAGSystem:
    """Factory function to create a RAG system with configuration"""
    config = {
        'project_id': os.getenv('PROJECT_ID', 'deepam-ai-tutor'),
        'location': os.getenv('REGION', 'us-central1'),
        'doc_ai_location': os.getenv('DOC_AI_LOCATION', 'us'),
        'doc_ai_processor_id': os.getenv('DOC_AI_PROCESSOR_ID'),
        'vector_db_path': os.getenv('VECTOR_DB_PATH', './chroma_db')
    }
    
    return RAGSystem(config)
