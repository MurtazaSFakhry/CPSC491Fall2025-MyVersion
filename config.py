import os
from dotenv import load_dotenv

# Load variables from .env in project root
load_dotenv()

def get_api_key():
    """Fetch OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Did you create a .env file?")
    return api_key
#def get_openai_key():
 #   return os.getenv("OPENAI_API_KEY")

def get_serpapi_key():
        """Return the SerpAPI key (supports both SERPAPI_API_KEY and SERPAPI_KEY).
        Priority order:
            1. SERPAPI_API_KEY
            2. SERPAPI_KEY (alternate naming)
        Returns None if neither set.
        """
        return os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")

def get_chroma_persist_path():
    return os.getenv("CHROMA_PERSIST_PATH", "./chroma_fcc_storage")

def get_collection_name():
    return os.getenv("CHROMA_COLLECTION_NAME", "fcc_documents")