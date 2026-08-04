import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
DATABASE_PATH = DATA_DIR / "sqlite.db"
OUTPUT_DIR = BASE_DIR / "outputs"
DETECTED_IMAGE_DIR = OUTPUT_DIR / "detected_images"
GENERATED_REPORT_DIR = OUTPUT_DIR / "generated_reports"
LOG_DIR = OUTPUT_DIR / "logs"
ASSET_DIR = BASE_DIR / "assets"

APP_NAME = "AI Multimodal Learning Assistant"
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
DEFAULT_EMBEDDING_MODEL = "bge-small-zh"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "compatible-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
