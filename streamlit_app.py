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
API_URL = os.environ.get('API_URL', 'http://localhost:8080')  # Default to local FastAPI server

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
    
    board = st.selectbox("Board", ["CBSE/NCERT", "Tamil Nadu Stateboard", "Karnataka Stateboard", "Andhra Pradesh Stateboard", "Maharashtra Stateboard"])
    state = board.split(" ")[0].lower() if "Stateboard" in board else "national"
    subject = st.selectbox("Subject", ["Science", "Math", "Social Studies", "English", "Regional Language"])
    language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi"])
    lang_code = ['en', 'hi', 'ta', 'te', 'kn', 'mr'][["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi"].index(language)]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    input_mode = st.radio("Input Mode", ("Text", "Voice"), key="input_mode")
    if input_mode == "Text":
        query = st.chat_input("Ask a question about your subject...")
    else:
        audio = audio_recorder.record()
        if audio:
            try:
                client = speech.SpeechClient()
                config = speech.RecognitionConfig(language_code=lang_code, enable_automatic_punctuation=True)
                response = client.recognize(config=config, audio=speech.Audio(content=audio))
                query = response.results[0].alternatives[0].transcript if response.results else ""
                st.write(f"Transcribed: {query}")
            except:
                st.error("Voice recognition failed. Try again or use text.")
                query = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        try:
            headers = {"Authorization": f"Bearer {st.session_state.user}"}
            chat_url = f"{API_URL}/chat?query={requests.utils.quote(query)}&class=10&board={board}&state={state}&subject={subject}&language={lang_code}"
            
            st.sidebar.markdown("### Chat Request Debug")
            st.sidebar.write("URL:", chat_url)
            st.sidebar.write("Headers:", headers)
            st.sidebar.write("Query:", query)
            
            response = requests.post(chat_url, headers=headers)
            
            st.sidebar.markdown("### Chat Response Debug")
            st.sidebar.write("Status Code:", response.status_code)
            try:
                st.sidebar.write("Response:", response.json())
            except:
                st.sidebar.write("Raw Response:", response.text)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({"role": "assistant", "content": data['response']})
                with st.chat_message("assistant"):
                    st.markdown(data['response'])
                    st.markdown("**Sources:** " + ", ".join(data['sources']))

                if st.checkbox("Play Response as Voice"):
                    tts_client = texttospeech.TextToSpeechClient()
                    synthesis_input = texttospeech.SynthesisInput(text=data['response'])
                    voice = texttospeech.VoiceSelectionParams(language_code=lang_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
                    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.0)
                    tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
                    st.audio(tts_response.audio_content, format="audio/mp3")

                with st.form("feedback"):
                    rating = st.slider("Rate this response (1-5)", 1, 5, key="rating")
                    comment = st.text_input("Feedback (optional)", key="comment")
                    if st.form_submit_button("Submit Feedback"):
                        requests.post(f"{API_URL}/feedback", json={"query_id": data['query_id'], "rating": rating, "comment": comment}, headers=headers)
                        st.success("Feedback submitted")
            else:
                st.error("Error: " + response.text)
        except Exception as e:
            st.error(f"Chat error: {str(e)}")