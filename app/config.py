import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DEMO_OUTPUTS_DIR = BASE_DIR / "demo_outputs"
SAMPLE_OUTPUTS_DIR = BASE_DIR / "sample_outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

for d in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, DEMO_OUTPUTS_DIR, SAMPLE_OUTPUTS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class Settings:
    PROJECT_NAME = "CloudClear AI"
    API_V1_STR = "/api"
    BASE_DIR = str(BASE_DIR)
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:saruu%401624@localhost:5432/cloudclear")
    SQLITE_FALLBACK_URL = "sqlite:///" + str(Path(__file__).resolve().parent.parent / "cloudclear.db")
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    OUTPUT_FOLDER = str(OUTPUT_DIR)
    MODELS_FOLDER = str(MODELS_DIR)
    DEMO_OUTPUTS_FOLDER = str(DEMO_OUTPUTS_DIR)
    SAMPLE_OUTPUTS_FOLDER = str(SAMPLE_OUTPUTS_DIR)
    CHECKPOINT_PATH = str(CHECKPOINT_DIR / "pix2pix_liss4_v1_best.pt")
    DEVICE = os.getenv("DEVICE", "cpu")

settings = Settings()