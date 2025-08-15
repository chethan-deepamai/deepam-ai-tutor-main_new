import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat?class=10&board=cbse&state=national&subject=Science&language=en", json={"query": "What is photosynthesis?"}, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert "response" in response.json()
    assert "sources" in response.json()
