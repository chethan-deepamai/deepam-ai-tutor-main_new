from google import genai

client = genai.Client(
    vertexai=True,                 # ensure you're using Vertex
    project="deepam-ai-tutor",
    location="us-central1"         # <-- change from asia-south1
)

resp = client.models.generate_content(
    model="publishers/google/models/gemini-2.0-flash",
    contents="What's the largest planet in our solar system?"
)
print(resp.text)