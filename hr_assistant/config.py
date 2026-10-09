"""All settings for the app live here, in one place."""


import os
from dotenv import load_dotenv

load_dotenv()


def _is_placeholder(value: str | None) -> bool:
    """Return True if the value is blank or still contains a demo placeholder."""
    if value is None:
        return True
    cleaned = value.strip()
    if not cleaned:
        return True
    lower = cleaned.lower()
    placeholders = {
        "changeme",
        "demo",
        "example",
        "placeholder",
        "your_groq_api_key_here",
        "your_jina_api_key_here",
        "your_portkey_api_key_here",
        "your_qdrant_api_key_here",
        "your-qdrant-instance.example.com",
    }
    return lower in placeholders or lower.startswith("your_") or "example" in lower


## ENV VAR / SECRET - LLMS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# GATEWAY

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

# GUARD MODEL 

GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

# TRACING 

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")





## DEFINE PATH - DATA / VECTOR STORE 

DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")

## VECTORE STORES 

# IN MEMORY 
# persistent memory - vectors # 100gb - ingestion 
# cloud memory 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hr_policy")

## MODELS 
# LLM and EMBEDING MODEL 

LLM_MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

## CHUNK / TEXT SPLITTING CONFIG 

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

# RETRIVAL RESULTS 
TOP_K_RESULTS = 3


## SYSTEM INSTRUCTIONS 

SYSTEM_PROMPT = (
    "You are a friendly HR assistant. Always use the search_hr_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, say you don't know "
    "instead of guessing."
)


def has_valid_cloud_config() -> bool:
    """Return True only when all required cloud settings look usable."""
    required = [
        GROQ_API_KEY,
        JINA_API_KEY,
        QDRANT_URL,
        QDRANT_API_KEY,
        PORTKEY_API_KEY,
    ]
    return all(value is not None and not _is_placeholder(value) for value in required)


def check_api_keys() -> bool:
    """Return False in offline/demo mode instead of crashing when secrets are missing."""
    if has_valid_cloud_config():
        return True
    return False
