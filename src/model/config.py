import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

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
    },
    "claude": {
        "name": "claude",
        "key": ANTHROPIC_KEY,
        "base_url": "https://api.anthropic.com/v1/"
    }
}

def get_config_for_model(model):
    match model:
        case "gemini-3-pro-preview" | "gemini-3.1-pro-preview":
            return DEFAULT_CONFIGS["gemini"] | { "model": model }
        case "gpt-5.4" | "gpt-5.3" | "gpt-5.2" | "gpt-5-mini":
            return DEFAULT_CONFIGS["openai"] | { "model": model }
        case "claude-opus-4-6":
            return DEFAULT_CONFIGS["claude"] | { "model": model }