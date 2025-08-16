from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
import os
import uuid
import tempfile
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import texttospeech
import firebase_admin
from firebase_admin import auth, credentials
from rag_system import create_rag_system
import logging
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred_path = os.path.join(os.path.dirname(__file__), 'firebase-adminsdk.json')
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

# Initialize Google Cloud Storage client
try:
    storage_client = storage.Client()
    BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'deepam-input')
    OUTPUT_BUCKET_NAME = os.getenv('GCS_OUTPUT_BUCKET_NAME', 'deepam-output')
    print(f"Attempting to connect to GCS buckets: {BUCKET_NAME}, {OUTPUT_BUCKET_NAME}")
    
    # For now, skip bucket validation and assume they exist
    # The actual bucket access will be tested during upload/download operations
    print(f"GCS client initialized with buckets: {BUCKET_NAME} (input), {OUTPUT_BUCKET_NAME} (output)")
    
except Exception as e:
    print(f"Warning: Could not initialize GCS client: {e}")
    print("Falling back to local storage...")
    storage_client = None
    BUCKET_NAME = None
    OUTPUT_BUCKET_NAME = None

security = HTTPBearer(auto_error=False)

# Initialize Text-to-Speech client
try:
    tts_client = texttospeech.TextToSpeechClient()
    logger.info("Text-to-Speech client initialized successfully")
except Exception as e:
    logger.warning(f"Text-to-Speech client initialization failed: {e}")
    tts_client = None

# Initialize RAG system
try:
    rag_system = create_rag_system()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    rag_system = None

app = FastAPI(title="DeepAM AI Tutor API", version="1.0.0")

# Pydantic models
class TTSRequest(BaseModel):
    text: str
    language_code: str = "en-US"
    voice_gender: str = "NEUTRAL"  # NEUTRAL, MALE, FEMALE
    speaking_rate: float = 1.0
    audio_encoding: str = "MP3"  # MP3, LINEAR16, OGG_OPUS

class ChatRequest(BaseModel):
    query: str
    class_: str = Field("10", alias="class")
    board: str = "CBSE/NCERT"
    state: str = "national"
    subject: str = "Science"
    language: str = "en"
    user_id: str


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase token"""
    if not credentials:
        return None
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

@app.get("/")
def root():
    return {"message": "Server is running"}

@app.get("/test")
def test():
    return {"message": "Test endpoint working"}

@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest, user=Depends(verify_token)):
    """
    Convert text to speech using Google Cloud Text-to-Speech
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not tts_client:
        raise HTTPException(status_code=503, detail="Text-to-Speech service not available")
    
    try:
        # Prepare the synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=request.text)
        
        # Configure voice selection
        voice_gender_map = {
            "NEUTRAL": texttospeech.SsmlVoiceGender.NEUTRAL,
            "MALE": texttospeech.SsmlVoiceGender.MALE,
            "FEMALE": texttospeech.SsmlVoiceGender.FEMALE
        }
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=request.language_code,
            ssml_gender=voice_gender_map.get(request.voice_gender.upper(), texttospeech.SsmlVoiceGender.NEUTRAL)
        )
        
        # Configure audio output
        encoding_map = {
            "MP3": texttospeech.AudioEncoding.MP3,
            "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
            "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS
        }
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=encoding_map.get(request.audio_encoding.upper(), texttospeech.AudioEncoding.MP3),
            speaking_rate=request.speaking_rate
        )
        
        # Perform the text-to-speech synthesis
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Return the audio content as a response
        media_type = "audio/mpeg" if request.audio_encoding.upper() == "MP3" else "audio/wav"
        return Response(
            content=response.audio_content,
            media_type=media_type,
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
        
    except Exception as e:
        logger.error(f"Text-to-speech conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    board: str = Form(...),
    subject: str = Form(...),
    class_: str = Form(...),
    language: str = Form(...),
    user: dict = Depends(verify_token)
):
    """
    Upload a file endpoint with Google Cloud Storage integration and RAG processing
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        
        if storage_client and BUCKET_NAME:
            try:
                # Upload to Google Cloud Storage
                bucket = storage_client.bucket(BUCKET_NAME)
                
                # Generate unique filename
                file_id = str(uuid.uuid4())
                gcs_filename = f"textbooks/{board}/{subject}/class_{class_}/{language}/{file_id}_{file.filename}"
                
                blob = bucket.blob(gcs_filename)
                blob.upload_from_string(content, content_type="application/pdf")
                
                gcs_uri = f"gs://{BUCKET_NAME}/{gcs_filename}"
                
                # Store metadata
                metadata = {
                    "board": board,
                    "subject": subject,
                    "class": class_,
                    "language": language,
                    "user_id": user.get("uid"),
                    "original_filename": file.filename,
                    "gcs_uri": gcs_uri,
                    "filename": file.filename
                }
                
                # Process document with RAG system
                if rag_system:
                    try:
                        logger.info(f"Processing document with RAG system: {file.filename}")
                        doc_id = rag_system.process_and_store_document(content, metadata)
                        logger.info(f"Document processed successfully: {doc_id}")
                        
                        return {
                            "message": "File uploaded and processed successfully",
                            "gcs_uri": gcs_uri,
                            "filename": file.filename,
                            "document_id": doc_id,
                            "metadata": metadata,
                            "size": len(content),
                            "processing_status": "completed"
                        }
                    except Exception as e:
                        logger.error(f"RAG processing failed: {e}")
                        return {
                            "message": "File uploaded but processing failed",
                            "gcs_uri": gcs_uri,
                            "filename": file.filename,
                            "metadata": metadata,
                            "size": len(content),
                            "processing_status": "failed",
                            "processing_error": str(e)
                        }
                else:
                    return {
                        "message": "File uploaded successfully (RAG processing unavailable)",
                        "gcs_uri": gcs_uri,
                        "filename": file.filename,
                        "metadata": metadata,
                        "size": len(content),
                        "processing_status": "rag_unavailable"
                    }
            except Exception as gcs_error:
                logger.warning(f"GCS upload failed: {gcs_error}. Falling back to local storage.")
                # Fall through to local storage
        else:
            # Fallback: save locally if GCS is not available
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
            
            with open(local_path, "wb") as f:
                f.write(content)
            
            # Mock GCS URI for local development
            gcs_uri = f"file://{local_path}"
            
            # Store metadata
            metadata = {
                "board": board,
                "subject": subject,
                "class": class_,
                "language": language,
                "user_id": user.get("uid"),
                "original_filename": file.filename,
                "gcs_uri": gcs_uri,
                "filename": file.filename,
                "local_path": local_path
            }
            
            # Process document with RAG system
            if rag_system:
                try:
                    logger.info(f"Processing local document with RAG system: {file.filename}")
                    doc_id = rag_system.process_and_store_document(content, metadata)
                    logger.info(f"Local document processed successfully: {doc_id}")
                    
                    return {
                        "message": "File uploaded and processed successfully (local storage)",
                        "gcs_uri": gcs_uri,
                        "filename": file.filename,
                        "document_id": doc_id,
                        "content_type": file.content_type,
                        "size": len(content),
                        "local_path": local_path,
                        "processing_status": "completed"
                    }
                except Exception as e:
                    logger.error(f"Local RAG processing failed: {e}")
                    return {
                        "message": "File uploaded but processing failed (local storage)",
                        "gcs_uri": gcs_uri,
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": len(content),
                        "local_path": local_path,
                        "processing_status": "failed",
                        "processing_error": str(e)
                    }
            else:
                return {
                    "message": "File uploaded successfully (local storage, RAG processing unavailable)",
                    "gcs_uri": gcs_uri,
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content),
                    "local_path": local_path,
                    "processing_status": "rag_unavailable"
                }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(verify_token)):
    """
    Chat endpoint for AI tutor with full RAG implementation
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        if rag_system:
            # Use RAG system to generate intelligent response
            logger.info(f"Processing query with RAG: {request.query}")
            result = rag_system.generate_answer(
                query=request.query,
                board=request.board,
                subject=request.subject,
                class_=request.class_,
                language=request.language,
                user_id=user.get("uid")
            )
            
            logger.info(f"RAG response generated with {result.get('context_used', 0)} context documents")
            return result
        else:
            # Fallback response when RAG is not available
            query_id = str(uuid.uuid4())
            response_text = f"I apologize, but the AI tutoring system is currently not available. Your question about '{request.query}' for {request.board} {request.subject} (Class {request.class_}) has been noted. Please try again later or contact support."
            
            return {
                "response": response_text,
                "sources": [],
                "query_id": query_id,
                "metadata": {
                    "board": request.board,
                    "subject": request.subject,
                    "class": request.class_,
                    "language": request.language,
                    "state": request.state
                },
                "context_used": 0,
                "confidence": "low",
                "status": "rag_unavailable"
            }
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        query_id = str(uuid.uuid4())
        return {
            "response": f"I encountered an error while processing your question about '{request.query}'. Please try rephrasing your question or try again later.",
            "sources": [],
            "query_id": query_id,
            "metadata": {
                "board": request.board,
                "subject": request.subject,
                "class": request.class_,
                "language": request.language,
                "state": request.state
            },
            "error": str(e),
            "context_used": 0,
            "confidence": "low"
        }

@app.post("/feedback")
async def submit_feedback(
    query_id: str = Form(...),
    rating: int = Form(...),
    comment: str = Form(""),
    user: dict = Depends(verify_token)
):
    """
    Submit feedback for a chat response
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Store feedback (in a real implementation, you'd save this to a database)
        feedback_data = {
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "user_id": user.get("uid"),
            "timestamp": "mock_timestamp"
        }
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": str(uuid.uuid4())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.post("/process")
async def process_document(
    gcs_input_uri: str = Form(...),
    lang_hints: list = Form(default=['en']),
    lang_primary: str = Form(default="en"),
    edition_year: int = Form(default=2024),
    overrides: dict = Form(default={})
):
    """
    Process document using Document AI (mock implementation)
    """
    try:
        # This would normally use Google Document AI to process the PDF
        # For now, return a mock response
        return {
            "message": "Document processing started",
            "gcs_input_uri": gcs_input_uri,
            "status": "processing",
            "job_id": str(uuid.uuid4())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
