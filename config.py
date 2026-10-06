import os
from pathlib import Path
import torch
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).parent
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.getenv("DATA_DIR") or BASE / "data")
IMAGES_DIR = DATA_DIR / "images"
KB_PATH = DATA_DIR / "kb.json"
SEED_PATH = Path(os.getenv("SEED_PATH", BASE / "seed_texts.json"))
TOP_K = int(os.getenv("TOP_K", "3"))
ALPHA = float(os.getenv("ALPHA", "0.6"))          # image->text vs caption->text
LOGIT_SCALE = float(os.getenv("LOGIT_SCALE", "50"))  # softmax sharpness for relevance %
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
