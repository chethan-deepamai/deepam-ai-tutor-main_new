# DeepAM AI Tutor - Core Learning Engine MVP

This is the MVP for DeepAM AI Tutor, a browser-based tutoring system for 10th-grade students. It processes PDF textbooks (CBSE/NCERT, Tamil Nadu, Karnataka, Andhra Pradesh, Maharashtra state boards), extracts text via OCR, and provides Socratic, localized explanations in 6 languages (English, Hindi, Tamil, Telugu, Kannada, Marathi) using text or voice input/output.

## Features
- Upload and process 10th-grade textbooks (Science, Math, Social Studies, English, Regional Language).
- RAG-powered search for textbook content, filtered by board/state/subject/language.
- Socratic tutoring: Step-by-step explanations, guiding questions, state-specific examples (e.g., Pongal for Tamil Nadu, farming for Karnataka).
- Streamlit UI with Firebase login/signup/invite, text/voice input, feedback button.
- GCP-based: Document AI, BigQuery, Vertex AI, Translate, Speech APIs, Cloud Run, Pub/Sub.

## Setup
1. Clone repo: `git clone https://github.com/yourusername/deepam-ai-tutor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up GCP (see Phase 3): Project, APIs, Firebase, buckets.
4. Deploy with Cloud Build (see `cloudbuild.yaml`).

## Testing
- Upload sample PDF (e.g., NCERT 10th Science).
- Test queries: “What is photosynthesis?” in English/Tamil.
- Invite 100 users via Firebase, collect feedback.

## Next Steps
- Phase 3: GCP setup (project, APIs, buckets).
- Future: Add languages, open-source LLM, mobile app.

## Issues
Use GitHub Issues to track tasks (e.g., “Test Tamil voice”, “Fix low-quality OCR chunks”).
