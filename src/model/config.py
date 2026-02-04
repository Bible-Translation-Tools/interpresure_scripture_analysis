import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_CONFIGS = {
    "gemini": {
        "name": "gemini",
        "key": GEMINI_KEY,
        "base_url": GOOGLE_BASE_URL,
    },
    "openai": {
        "name": "openai",
        "key": OPENAI_KEY,
        "base_url": None,
    }
}

def get_config_for_model(model):
    match model:
        case "gemini-3-pro-preview":
            return DEFAULT_CONFIGS["gemini"] | { "model": model }
        case "gpt-5.2" | "gpt-5-mini":
            return DEFAULT_CONFIGS["openai"] | { "model": model }