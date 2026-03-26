import os

# Existing settings
pytesseract_exe_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
SETTINGS = ['BASE_DIR', 'pytesseract_exe_path', 'tessdata_path', 'reset_data_boolean', 'use_llm',
            'CHUNK_SIZE', 'CHUNK_OVERLAP', 'EMBEDDING_MODEL_NAME', 'LLM_MODEL_NAME',
            'USE_LOCAL_LLM', 'HF_API_TOKEN', 'GENERATION_TEMPERATURE', 'REFORMULATION_TEMPERATURE', 'TOP_K_PERCENT', 'SCORE_THRESHOLD']
SUPPORTED_FORMATS = ['rtf', 'pdf', 'docx', 'txt', 'doc']

BASE_DIR = r'C:\Users\Timofey'  # default, can be changed via settings
reset_data_boolean = False
use_llm = True

# New RAG settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v2-moe'  # or 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'microsoft/Phi-3-mini-4k-instruct'       # choose your model
USE_LOCAL_LLM = True
HF_API_TOKEN = ''  # set if using remote
GENERATION_TEMPERATURE = 0.7
REFORMULATION_TEMPERATURE = 0.1

# Paths
CHROMA_PATH = "chroma"
BM25_RETRIEVER_PATH = "appdata"
# Retrieval parameters
TOP_K_PERCENT = 0.05  # Retrieve 5% of total documents
SCORE_THRESHOLD = 0.2  # Minimum similarity score for vector results