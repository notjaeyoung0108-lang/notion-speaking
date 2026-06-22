"""Config for the speaking sentence + comic pipeline."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("Elevenlabs_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_SPEAKING_DATABASE_ID = os.getenv("NOTION_SPEAKING_DATABASE_ID")
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

_override = os.getenv("PIPELINE_DATE")
_now = datetime.strptime(_override, "%y.%m.%d") if _override else datetime.now()
YY_MM = _now.strftime("%y.%m")
MM_DD = _now.strftime("%m.%d")
YY_MM_DD = _now.strftime("%y.%m.%d")
TODAY_DATE = _now.strftime("%Y-%m-%d")

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
SENTENCE_DIR = DATA_DIR / "sentences" / YY_MM
COMIC_DAY_DIR = DATA_DIR / "comic_out" / YY_MM / MM_DD
TTS_DAY_DIR = DATA_DIR / "TTS" / YY_MM / MM_DD
AAC_DAY_DIR = DATA_DIR / "AAC" / YY_MM

for d in (SENTENCE_DIR, COMIC_DAY_DIR, TTS_DAY_DIR, AAC_DAY_DIR):
    d.mkdir(parents=True, exist_ok=True)

STRUCTURED_CSV = SENTENCE_DIR / f"structured_{YY_MM_DD}.csv"
CLEAN_CSV = SENTENCE_DIR / f"{YY_MM_DD}_speaking.csv"
HISTORY_PATH = DATA_DIR / "sentence_history.txt"
FINAL_AAC_ENG = AAC_DAY_DIR / f"{YY_MM_DD}_speaking_영어.aac"

DEFAULT_THEME = os.getenv(
    "SPEAKING_THEME",
    "daily and workplace situations a Korean B1-B2 learner is likely to speak in",
)
DEFAULT_COUNT = int(os.getenv("SPEAKING_COUNT", "20"))

MODAL_DISABLED = os.getenv("MODAL_DISABLED") == "1"
BUILD_AAC = os.getenv("BUILD_AAC") == "1"
