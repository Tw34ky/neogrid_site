import os

# Existing settings
pytesseract_exe_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
SETTINGS = ['BASE_DIR', 'pytesseract_exe_path', 'tessdata_path', 'reset_data_boolean', 'use_llm']
CHUNK_SIZE = r'600'
GENERATION_TEMPERATURE = r'0.7'
SUPPORTED_FORMATS = ['rtf', 'pdf', 'docx', 'txt', 'doc']

BASE_DIR = r'C:\Users\Timofey'  # default, can be changed via settings
reset_data_boolean = False
use_llm = True

# New RAG settings
CHUNK_SIZE = r'600'
CHUNK_OVERLAP = r'100'
EMBEDDING_MODEL_NAME = 'nomic-ai/nomic-embed-text-v2-moe'  # or 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'microsoft/Phi-3-mini-4k-instruct'       # choose your model
USE_LOCAL_LLM = r'on'
HF_API_TOKEN = ''  # set if using remote
GENERATION_TEMPERATURE = r'0.7'
REFORMULATION_TEMPERATURE = r'0.1'

# Paths
CHROMA_PATH = "chroma"
BM25_RETRIEVER_PATH = "appdata"
# Retrieval parameters
TOP_K_PERCENT = r'0.05'
SCORE_THRESHOLD = r'0.2'
