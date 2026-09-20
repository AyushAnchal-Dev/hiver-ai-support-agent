"""
Application Configuration and Constants
"""
import os
from pathlib import Path

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = BASE_DIR / "archive"
REPORT_DIR = BASE_DIR / "report"

# Primary Files
RAW_TWCS_CSV = ARCHIVE_DIR / "twcs.csv"
SPOTIFY_RAW_CSV = DATA_DIR / "spotify_raw.csv"
SPOTIFY_THREADS_CSV = DATA_DIR / "spotify_threads.csv"
SPOTIFY_CONVERSATIONS_JSON = DATA_DIR / "spotify_conversations.json"
SPOTIFY_METADATA_JSON = DATA_DIR / "spotify_metadata.json"

# Brand Parameters
TARGET_BRAND = "SpotifyCares"
TARGET_BRAND_LOWER = "spotifycares"
TARGET_HANDLE = "@SpotifyCares"

# Data Cleaning Regular Expressions
USER_HANDLE_REGEX = r"@\d+"
URL_REGEX = r"https?://\S+|www\.\S+"
AGENT_SIGNATURE_REGEX = r"(\s*[\^–-][A-Za-z]{2,4}\s*(\(\d+/\d+\))?|\s*\^\w+|\s*-[A-Za-z]{2,3})$"
DM_KEYWORDS_REGEX = r"\b(dm|direct message|private message)\b"

# Conversation Horizon Limits
MAX_CONTEXT_TURNS = 4
MIN_TEXT_LENGTH = 3

# Stage 6A Settings Loader Re-export
from app.settings import Settings, get_settings

