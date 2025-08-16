import streamlit as st
import requests
import os
import json
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
import audio_recorder_streamlit as audio_recorder
import firebase_admin
from firebase_admin import auth, credentials

# Firebase configuration
API_URL = os.environ.get('API_URL', 'http://localhost:8080')  # Default API server

# Load Firebase Admin credentials
cred_path = os.path.join(os.path.dirname(__file__), 'firebase-adminsdk.json')
with open(cred_path) as f:
    cred_json = json.load(f)

FIREBASE_CONFIG = {
    "apiKey": os.environ.get('FIREBASE_WEB_API_KEY', 'AIzaSyD5_qcruyb-gVyk5KDqK8ec894qSovUJjU'),
    "authDomain": f"{cred_json['project_id']}.firebaseapp.com",
    "projectId": cred_json['project_id'],
}
FIREBASE_AUTH_DOMAIN = "https://identitytoolkit.googleapis.com/v1/accounts"
FIREBASE_WEB_API_KEY = FIREBASE_CONFIG["apiKey"]

# Debug info
st.sidebar.write("Debug Info:")
st.sidebar.write(f"Project ID: {FIREBASE_CONFIG['projectId']}")
st.sidebar.write(f"Auth Domain: {FIREBASE_CONFIG['authDomain']}")
st.sidebar.json(cred_json)

if not firebase_admin._apps:
    cred_path = os.path.join(os.path.dirname(__file__), 'firebase-adminsdk.json')
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

st.set_page_config(page_title="DeepAM AI Tutor", layout="wide")
st.markdown("""
<style>
    .stApp { max-width: 800px; margin: auto; }
    .stChatMessage { padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #d1e7dd; }
    .stChatMessage.assistant { background-color: #f8d7da; }
    [role="radiogroup"] { margin-bottom: 20px; }
    @media (max-width: 600px) { .stApp { padding: 10px; } }
</style>
""", unsafe_allow_html=True)

st.title("DeepAM AI Tutor - 10th Grade MVP")
st.markdown("Learn with simple, local explanations in your language! (Privacy: Your queries are anonymized.)")

def debug_request(method, url, headers, params, data):
    st.sidebar.markdown("### Debug Request")
    st.sidebar.write(f"Method: {method}")
    st.sidebar.write(f"URL: {url}")
    st.sidebar.write("Headers:", headers)
    st.sidebar.write("Params:", params)
    st.sidebar.write("Data:", data)

def debug_response(response):
    st.sidebar.markdown("### Debug Response")
    st.sidebar.write(f"Status Code: {response.status_code}")
    st.sidebar.write(f"Response Headers: {dict(response.headers)}")
    try:
        st.sidebar.write("Response JSON:", response.json())
    except:
        st.sidebar.write("Response Text:", response.text)

def sign_in_with_email_and_password(email, password):
    request_data = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    url = f"{FIREBASE_AUTH_DOMAIN}:signInWithPassword"
    
    headers = {
        "Content-Type": "application/json"
    }
    params = {"key": FIREBASE_WEB_API_KEY}
    
    debug_request("POST", url, headers, params, request_data)
    
    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=request_data
        )
        response.raise_for_status()
        debug_response(response)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication error: {str(e)}")
        if response:
            st.error(f"Response content: {response.text}")
        return None

def create_user_with_email_and_password(email, password):
    request_data = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    url = f"{FIREBASE_AUTH_DOMAIN}:signUp"
    
    headers = {
        "Content-Type": "application/json"
    }
    params = {"key": FIREBASE_WEB_API_KEY}
    
    debug_request("POST", url, headers, params, request_data)
    
    try:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=request_data
        )
        response.raise_for_status()
        debug_response(response)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication error: {str(e)}")
        if response:
            st.error(f"Response content: {response.text}")
        return None

if 'user' not in st.session_state:
    page = st.sidebar.selectbox("Auth", ["Login", "Signup"])
    if page == "Signup":
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Signup"):
            try:
                result = create_user_with_email_and_password(email, password)
                if result and 'idToken' in result:
                    st.session_state.user = result['idToken']
                    st.success("Signed up! You can now use the tutor.")
                elif result and 'error' in result:
                    st.error(f"Error signing up: {result.get('error', {}).get('message', 'Unknown error')}")
                else:
                    st.error("Signup failed. Please try again.")
            except Exception as e:
                st.error(f"Error signing up: {str(e)}")
    else:  # Login
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            try:
                result = sign_in_with_email_and_password(email, password)
                if result and 'idToken' in result:
                    st.session_state.user = result['idToken']
                    st.success("Logged in successfully!")
                elif result and 'error' in result:
                    st.error(f"Invalid credentials: {result.get('error', {}).get('message', 'Unknown error')}")
                else:
                    st.error("Authentication failed. Please check your credentials.")
            except Exception as e:
                st.error(f"Invalid credentials: {str(e)}")

if 'user' in st.session_state:
    st.sidebar.button("Logout", on_click=lambda: st.session_state.pop('user', None))
    
    # File Upload Section
    st.header("📚 Upload Textbook")
    st.write("Debug: PDF upload interface loaded")
    st.markdown("Upload your textbook PDF to add it to the knowledge base for personalized learning.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            board_upload = st.selectbox("Board", ["CBSE/NCERT", "Tamil Nadu Stateboard", "Karnataka Stateboard", "Andhra Pradesh Stateboard", "Maharashtra Stateboard"], key="upload_board")
        with col2:
            subject_upload = st.selectbox("Subject", ["Science", "Math", "Social Studies", "English", "Regional Language"], key="upload_subject")
        
        class_upload = st.selectbox("Class", ["10", "9", "11", "12"], key="upload_class")
        language_upload = st.selectbox("Language", ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi"], key="upload_language")
        
        if st.button("Upload and Process"):
            with st.spinner("Uploading and processing your textbook..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {
                        "board": board_upload,
                        "subject": subject_upload,
                        "class_": class_upload,
                        "language": language_upload
                    }
                    headers = {"Authorization": f"Bearer {st.session_state.user}"}
                    
                    response = requests.post(f"{API_URL}/upload", files=files, data=data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Textbook uploaded successfully to {result['gcs_uri']}!")
                        st.info("Your textbook is queued for processing. You can ask questions about it soon.")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        st.sidebar.write(f"Error Details: {response.text}")
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")
                    st.sidebar.write(f"Exception: {str(e)}")
    
    st.divider()

    # Chat Section
    st.header("💬 AI Tutor Chat")

    # --- Callback functions for state management ---
    def handle_clear_cache(audio_key, tts_key):
        """Callback to clear audio cache and reset TTS checkbox."""
        if audio_key in st.session_state:
            del st.session_state[audio_key]
        if tts_key in st.session_state:
            st.session_state[tts_key] = False
        st.success("Audio cache cleared!")

    def handle_tts_error(key):
        """Callback to safely reset a TTS checkbox on error."""
        if key in st.session_state:
            st.session_state[key] = False

    board = st.selectbox("Board", ["CBSE/NCERT", "Tamil Nadu Stateboard", "Karnataka Stateboard", "Andhra Pradesh Stateboard", "Maharashtra Stateboard"])
    state = board.split(" ")[0].lower() if "Stateboard" in board else "national"
    subject = st.selectbox("Subject", ["Science", "Math", "Social Studies", "English", "Regional Language"])
    language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi"])
    lang_code_map = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", 
        "Kannada": "kn", "Marathi": "mr"
    }
    lang_code = lang_code_map.get(language, "en")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                tts_key = f"tts_enabled_{i}"
                audio_key = f"audio_content_{i}"
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.checkbox(
                        "🔊 Play Response as Sound", 
                        key=tts_key,
                        help="Check this box to generate and play audio of the response"
                    )
                
                with col2:
                    st.button(
                        "🗑️ Clear Cache", 
                        key=f"clear_cache_{i}",
                        on_click=handle_clear_cache,
                        args=(audio_key, tts_key)
                    )

                if tts_key in st.session_state and st.session_state[tts_key]:
                    if audio_key not in st.session_state:
                        with st.spinner("🎵 Generating audio..."):
                            try:
                                headers = {"Authorization": f"Bearer {st.session_state.user}"}
                                tts_response = requests.post(
                                    f"{API_URL}/text-to-speech",
                                    headers=headers,
                                    json={
                                        "text": message["content"],
                                        "language_code": lang_code,
                                    }
                                )
                                if tts_response.status_code == 200:
                                    st.session_state[audio_key] = tts_response.content
                                    st.success("✅ Audio generated successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ TTS failed: {tts_response.text}")
                                    handle_tts_error(tts_key)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ TTS failed: {str(e)}")
                                handle_tts_error(tts_key)
                                st.rerun()
                    
                    if audio_key in st.session_state:
                        st.audio(st.session_state[audio_key], format="audio/mp3")

    # Handle user input
    query = None
    input_mode = st.radio("Input Mode", ("Text", "Voice"), key="input_mode")
    if input_mode == "Text":
        query = st.chat_input("Ask a question about your subject...")
    else:
        audio_bytes = audio_recorder(key='audio_input')
        if audio_bytes:
            st.info("Transcribing audio...")
            try:
                client = speech.SpeechClient()
                audio = speech.RecognitionAudio(content=audio_bytes)
                config = speech.RecognitionConfig(
                    language_code=lang_code,
                    enable_automatic_punctuation=True
                )
                response = client.recognize(config=config, audio=audio)
                if response.results:
                    query = response.results[0].alternatives[0].transcript
                    st.write(f"Transcribed: *{query}*")
                else:
                    st.warning("Could not transcribe audio. Please try again.")
            except Exception as e:
                st.error(f"Voice recognition failed: {e}")

    # Process new query
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            with st.spinner("DeepAM is thinking..."):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.user}"}
                    chat_url = f"{API_URL}/chat"
                    
                    payload = {
                        "query": query,
                        "class": "10",
                        "board": board,
                        "state": state,
                        "subject": subject,
                        "language": lang_code,
                        "user_id": st.session_state.user
                    }

                    response = requests.post(chat_url, headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Check for backend error details before displaying response
                        if "error_details" in data or data.get("confidence") == "error":
                            st.error(f"An error occurred on the backend: {data.get('error_details', 'No details provided.')}")
                        else:
                            assistant_response = data.get('response', "Sorry, I couldn't generate a response.")
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                            st.rerun()
                    else:
                        st.error(f"Error from chat API: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"An error occurred while contacting the chat service: {e}")
        
    # Feedback form
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        latest_response = st.session_state.messages[-1]["content"]
        query_id = hash(latest_response)

        with st.form(key=f"feedback_form_{query_id}"):
            st.write("Was this response helpful?")
            rating = st.slider("Rate this response (1=Not helpful, 5=Very helpful)", 1, 5, 3, key=f"rating_{query_id}")
            comment = st.text_input("Feedback (optional)", key=f"comment_{query_id}")
            
            if st.form_submit_button("Submit Feedback"):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.user}"}
                    feedback_payload = {
                        "query_id": str(query_id), 
                        "rating": rating, 
                        "comment": comment
                    }
                    requests.post(f"{API_URL}/feedback", json=feedback_payload, headers=headers)
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error(f"Could not submit feedback: {e}")
