import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

LLM_MODEL = "llama-3.3-70b-versatile" # llama gelecek

GROQ_API_KEY=os.getenv("GROQ_API_KEY")