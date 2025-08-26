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
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import vertexai
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

class VertexAISearchStore:
    """Handles vector storage and retrieval using Vertex AI Search and GCS."""

    def __init__(self, project_id: str, location: str, gcs_bucket_name: str, index_id: str, endpoint_id: str, deployed_index_id: str, api_endpoint: str):
        self.project_id = project_id
        self.location = location
        self.gcs_bucket_name = gcs_bucket_name
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.deployed_index_id = deployed_index_id
        self.api_endpoint = api_endpoint
        
        vertexai.init(project=project_id, location=location)
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(gcs_bucket_name)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.index = self._get_or_create_index()
        
        # Initialize the MatchServiceClient
        client_options = {"api_endpoint": self.api_endpoint}
        self.vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)
        
        # Construct the full Index Endpoint resource name
        self.index_endpoint_name = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_id}"

    def _get_or_create_index(self):
        """Gets or creates a Vertex AI Search Index."""
        try:
            index_path = f"projects/{self.project_id}/locations/{self.location}/indexes/{self.index_id}"
            logger.info(f"Using Vertex AI Index path: {index_path}")
            return aiplatform.MatchingEngineIndex(index_name=index_path)
        except Exception as e:
            logger.error(f"Error getting Vertex AI Index: {e}")
            # In a real-world scenario, you might want to create it here.
            # For now, we re-raise to make the configuration error clear.
            raise Exception(f"Failed to initialize Vertex AI Index with ID '{self.index_id}'. Please ensure it exists.") from e

    def _get_or_create_endpoint(self):
        """Gets or creates a Vertex AI Index Endpoint."""
        # This method is no longer needed as we are using MatchServiceClient directly
        # and constructing the endpoint name in __init__
        pass

    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Adds a document to Vertex AI Search and stores metadata in GCS."""
        chunks = self.text_splitter.split_text(text)
        doc_id = str(uuid.uuid4())
        
        embeddings = []
        file_data_to_upload = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedding_model.encode(chunk).tolist()
            
            chunk_metadata = {
                "board": metadata.get('board', ''),
                "subject": metadata.get('subject', ''),
                "class": metadata.get('class', ''),
                "language": metadata.get('language', ''),
                "filename": metadata.get('filename', ''),
                "text": chunk,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "document_id": doc_id
            }
            
            # Store metadata and text chunk in GCS
            blob = self.bucket.blob(f"{chunk_id}.json")
            blob.upload_from_string(json.dumps(chunk_metadata, indent=2), content_type='application/json')

            embeddings.append({
                "datapoint_id": chunk_id,
                "feature_vector": embedding
            })

        # Upsert embeddings to Vertex AI Index
        # Note: The actual API for this might differ based on SDK version.
        # This is a conceptual representation.
        # The MatchingEngineIndex.upsert() method is not available in the SDK.
        # A common way is to batch data into a JSONL file and upload to GCS, then call index.update()
        # For simplicity, we'll log this action.
        logger.info(f"Prepared {len(embeddings)} embeddings for upsert to index {self.index.name}")
        # self.index.upsert(datapoints=embeddings) # This is conceptual
        
        # A more realistic approach:
        # 1. Create a JSONL file with embeddings.
        # 2. Upload to GCS.
        # 3. Call self.index.update(contents_delta_uri=...)
        # This is complex for a synchronous flow. For now, we log.
        try:
            logger.info(f"Upserting {len(embeddings)} embeddings to index {self.index.name}")
            self.index.upsert_datapoints(datapoints=embeddings)
            logger.info(f"Successfully upserted {len(embeddings)} embeddings.")
        except Exception as e:
            logger.error(f"Failed to upsert embeddings to Vertex AI Index: {e}")
            # Depending on the desired behavior, you might want to clean up GCS files here.
            raise Exception("Failed to add document to Vertex AI Index.") from e
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks. Metadata stored in GCS.")
        return doc_id
    
    def search_similar(self, query: str, board: str, subject: str, class_: str,
                       language: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content using Vertex AI Vector Search."""
        # Generate query embedding
        try:
            # Some embedding clients expose get_embeddings; SentenceTransformer does not.
            query_embedding = self.embedding_model.get_embeddings([query])[0].values  # type: ignore[attr-defined]
            logger.info(f"Query embedding shape: {len(query_embedding)}")
        except AttributeError:
            query_embedding = self.embedding_model.encode(query).tolist()
            logger.info(f"Query embedding shape (SentenceTransformers): {len(query_embedding)}")

        # Prepare datapoint
        datapoint = aiplatform_v1.IndexDatapoint(feature_vector=query_embedding)

        # Prepare request
        query_request = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=top_k
        )

        request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=self.index_endpoint_name,
            deployed_index_id=self.deployed_index_id,
            queries=[query_request],
            return_full_datapoint=True,
        )

        try:
            # Execute search
            response = self.vector_search_client.find_neighbors(request)
            logger.info(f"Find_neighbors response: {response}")

            similar_docs: List[Dict[str, Any]] = []
            if response.nearest_neighbors:
                for neighbor_list in response.nearest_neighbors:
                    for neighbor in neighbor_list.neighbors:
                        # Get ID of the matched datapoint
                        vector_id = getattr(neighbor.datapoint, 'datapoint_id', None) or getattr(neighbor, 'datapoint_id', None)
                        if not vector_id:
                            logger.warning("Neighbor missing datapoint_id; skipping")
                            continue

                        # Retrieve metadata from GCS
                        blob = self.bucket.blob(f"{vector_id}.json")
                        if not blob.exists():
                            logger.warning(f"Metadata file {vector_id}.json not found in GCS bucket {self.bucket.name}")
                            continue

                        metadata_string = blob.download_as_string()
                        doc_data = json.loads(metadata_string.decode('utf-8'))

                        # Optional metadata filtering
                        def _matches(field_key: str, field_value: str) -> bool:
                            return not field_value or (str(doc_data.get(field_key, '')).strip().lower() == str(field_value).strip().lower())

                        if not (_matches('board', board) and _matches('subject', subject) and _matches('class', class_) and _matches('language', language)):
                            continue

                        similarity = 1.0 - neighbor.distance if neighbor.distance is not None else 0.0
                        similar_docs.append({
                            "text": doc_data.get("text", ""),
                            "metadata": {k: v for k, v in doc_data.items() if k != "text"},
                            "similarity": similarity
                        })
            else:
                logger.warning("No nearest neighbors found in response")

            return similar_docs

        except Exception as e:
            logger.error(f"Vertex AI search failed: {e}")
            return []
    
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
            # Local embeddings for any semantic checks inside this class
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("AI Response Generator initialized (AI Studio key).")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.client = None
            self.model_name = None
            self.embedding_model = None
    def _is_textbook_question(self, query: str, context_docs: List[Dict[str, Any]], subject: str) -> bool:
        """Determine if the query is textbook-specific based on semantic similarity to context."""
        if not context_docs:
            return False
        
        # Encode query and context documents
        query_embedding = self.embedding_model.encode(query)
        context_texts = [doc['text'] for doc in context_docs]
        context_embeddings = self.embedding_model.encode(context_texts)
        
        # Calculate similarity scores
        similarities = np.dot(context_embeddings, query_embedding) / (
            np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        max_similarity = np.max(similarities) if similarities.size > 0 else 0.0
        
        # Use a dynamic baseline: compare against similarity to a neutral sentence (e.g., "This is a general topic")
        neutral_sentence = "This is a general topic"
        neutral_embedding = self.embedding_model.encode(neutral_sentence)
        neutral_similarities = np.dot(context_embeddings, neutral_embedding) / (
            np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(neutral_embedding)
        )
        avg_neutral_similarity = np.mean(neutral_similarities) if neutral_similarities.size > 0 else 0.0
        
        # Classify as textbook-specific if query similarity significantly exceeds the neutral baseline
        return max_similarity > (avg_neutral_similarity * 1.5)  # Dynamic multiplier for relative threshold

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client or not self.model_name:
            return {"response": "AI not configured. Check GEMINI_API_KEY.", "sources": [], "confidence": "error"}

        if not context_docs:
            msg = (f"I couldn't find specific information about '{query}' in the uploaded materials "
                   f"for {metadata.get('subject','the subject')}. Try rephrasing your question or upload the relevant textbook.")
            return {"response": msg, "sources": [], "confidence": "low"}

        combined_context = "\n\n".join([doc["text"] for doc in context_docs])
        prompt = f"""
Hi! I'm DeepAM, your friendly AI tutor for 10th-grade students. Ask me anything, and I’ll give you a clear, simple answer!

Student Details:
- Board: {metadata.get('board', 'CBSE/NCERT')}
- Subject: {metadata.get('subject', 'Science')}
- Language: {metadata.get('language', 'English')}

CONTEXT FROM TEXTBOOK:
---
{combined_context}
---

QUESTION: "{query}"

Instructions:
1. **Answer Directly**:
   - For questions about the subject ({metadata.get('subject', 'Science')}) or textbook-specific terms (e.g., 'chapter', 'topic', 'photosynthesis'), use ONLY the provided context and answer concisely.
   - For general questions (e.g., about topics outside the subject or textbook, like 'What is the tallest mountain?' or 'Who invented the telephone?'), answer using general knowledge, even if no context is provided.
2. **Handling Insufficient Context**:
   - If the question is textbook-related but the context is missing or insufficient, say so and suggest checking specific chapters or uploading the textbook.
3. **Tone and Style**:
   - Use a friendly, direct, and conversational tone, like a tutor chatting with a student.
   - Answer in {metadata.get('language', 'English')}.
   - Keep it simple, engaging, and encouraging for a 10th-grade student.
   - Avoid mentioning question types, instructions, or context unless necessary for textbook-related answers.
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
        self.vector_store = VertexAISearchStore(
            project_id=config.get('project_id'),
            location=config.get('location', 'us-central1'),
            gcs_bucket_name=config.get('gcs_bucket_name'),
            index_id=config.get('vertex_index_id'),
            endpoint_id=config.get('vertex_endpoint_id'),
            deployed_index_id=config.get('vertex_deployed_index_id'),
            api_endpoint=config.get('vertex_api_endpoint')
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
        'gcs_bucket_name': os.getenv('GCS_BUCKET_NAME'),
        'vertex_index_id': os.getenv('VERTEX_INDEX_ID'),
        'vertex_endpoint_id': os.getenv('VERTEX_ENDPOINT_ID'),
        'vertex_deployed_index_id': os.getenv('VECTOR_SEARCH_DEPLOYED_INDEX_ID'),
        'vertex_api_endpoint': os.getenv('VECTOR_SEARCH_API_ENDPOINT')
    }
    return RAGSystem(config)