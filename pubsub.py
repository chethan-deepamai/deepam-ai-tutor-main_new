from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import json
import os
import base64
import requests

app = FastAPI()

MAIN_SERVICE_URL = os.environ.get('MAIN_SERVICE_URL')
API_KEY = os.environ.get('API_KEY', '')

class PubSubMessage(BaseModel):
    message: dict
    subscription: str

@app.post("/pubsub")
def pubsub(event: PubSubMessage):
    try:
        message_data = base64.b64decode(event.message['data']).decode('utf-8')
        message = json.loads(message_data)
        if message.get('name', '').endswith('.pdf'):
            gcs_input_uri = f"gs://{message['bucket']}/{message['name']}"
            headers = {"X-API-KEY": API_KEY} if API_KEY else {}
            payload = {
                "gcs_input_uri": gcs_input_uri,
                "lang_hints": ['en', 'hi', 'ta', 'te', 'kn', 'mr'],
                "lang_primary": "en",
                "edition_year": 2024,
                "overrides": {}
            }
            response = requests.post(f"{MAIN_SERVICE_URL}/process", json=payload, headers=headers)
            response.raise_for_status()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PubSub error: {str(e)}")
