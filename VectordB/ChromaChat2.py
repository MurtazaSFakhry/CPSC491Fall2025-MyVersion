"""
Simple CLI chat that retrieves top-k documents from Pinecone and answers with inline citations.

Usage:
  python3 VectordB/ChromaChat2.py

Features:
- Uses Pinecone cloud vector storage for retrieval
- Uses same embedding model (text-embedding-3-small with 1536 dimensions)
- Markdown-formatted source list with titles and links
- Falls back to SerpAPI web search when Pinecone has no good matches
- Saves new external sources back to Pinecone for future queries
- Explicit guardrails to avoid hallucinating beyond sources
- Retries on transient API errors
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # parent of VectordB or ingestion
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import datetime
from typing import List, Dict, Tuple

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from chromadb import PersistentClient
from pinecone import Pinecone
from openai import OpenAI
try:
    from serpapi import GoogleSearch
except ImportError:
    # Newer serpapi versions use different import
    from serpapi.google_search import GoogleSearch
from uuid import uuid4
import numpy as np
import hashlib

# === Configuration ===

# Access API keys from Streamlit secrets or .env fallback
# Always try .env first for local development
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fcc-chatbot-index")
ID_STRATEGY = os.getenv("PINECONE_ID_STRATEGY", "url")  # 'url' (default) or 'content'

# Override with Streamlit secrets if available (for cloud deployment)
if STREAMLIT_AVAILABLE:
    try:
        if hasattr(st, 'secrets') and st.secrets:
            OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
            SERPAPI_API_KEY = st.secrets.get("SERPAPI_KEY", st.secrets.get("SERPAPI_API_KEY", SERPAPI_API_KEY))
            PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", PINECONE_API_KEY)
            PINECONE_INDEX = st.secrets.get("PINECONE_INDEX", PINECONE_INDEX)
    except:
        pass  # Use .env values

# Determine correct path for ChromaDB based on where script is running from
local_path = "./chroma_fcc_storage"
parent_path = "../chroma_fcc_storage"
if os.path.exists(local_path):
    PERSIST_PATH = local_path
elif os.path.exists(parent_path):
    PERSIST_PATH = parent_path
else:
    PERSIST_PATH = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_fcc_storage")

COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fcc_documents")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536  # Using 1536 dimensions for Pinecone compatibility
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
FALLBACK_TEXT = "No information available in the dataset or external sources for that question."
RELEVANCE_THRESHOLD = 0.35  # Cosine similarity threshold for topic relevance

# Reference topics for emergency alerting systems
EMERGENCY_TOPICS = [
    "emergency alert system EAS wireless emergency alerts WEA",
    "integrated public alert warning system IPAWS disaster response",
    "Federal Communications Commission FCC public safety communications",
    "emergency management FEMA cybersecurity policy national security",
    "emergency broadcast system disaster preparedness crisis communication",
    "public warning systems emergency notifications alert infrastructure",
    "emergency response protocols homeland security critical infrastructure"
]

# Ingestion-like params for saving external sources
MIN_ARTICLE_LENGTH = 200
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# === Clients ===

# Initialize Pinecone
if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not set in .env file")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX)

# Keep ChromaDB as backup (optional)
client = PersistentClient(path=PERSIST_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets or .env"""
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set in Streamlit secrets or .env file")
        sys.exit(1)
    return OpenAI(api_key=OPENAI_API_KEY)

openai_client = get_openai_client()

# Pre-compute embeddings for emergency topics (cached for efficiency)
_topic_embeddings_cache = None

def get_topic_embeddings() -> List[List[float]]:
    """Get or compute embeddings for emergency topics (cached)."""
    global _topic_embeddings_cache
    if _topic_embeddings_cache is None:
        #print("🔄 Computing reference embeddings for emergency topics...")
        _topic_embeddings_cache = [embed_text(topic) for topic in EMERGENCY_TOPICS]
    return _topic_embeddings_cache

# === Embedding & Retrieval ===

def embed_text(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIMENSIONS
    )
    return resp.data[0].embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def is_relevant_to_emergency_systems(query: str) -> Tuple[bool, float]:
    """
    Check if the query is relevant to emergency alerting systems using cosine similarity.
    Returns (is_relevant, max_similarity_score).
    """
    try:
        # Get query embedding
        query_embedding = embed_text(query)
        
        # Get pre-computed topic embeddings
        topic_embeddings = get_topic_embeddings()
        
        # Calculate similarity with each emergency topic
        similarities = [cosine_similarity(query_embedding, topic_emb) 
                       for topic_emb in topic_embeddings]
        
        # Get maximum similarity
        max_similarity = max(similarities)
        
        # Check if above threshold
        is_relevant = max_similarity >= RELEVANCE_THRESHOLD
        
        return is_relevant, max_similarity
        
    except Exception as e:
        print(f"⚠️ Relevance check error: {e}")
        # Default to allowing the question if check fails
        return True, 1.0

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embed helper to reduce API calls."""
    if not texts:
        return []
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMENSIONS
    )
    return [r.embedding for r in resp.data]

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    """Retrieve relevant chunks from Pinecone"""
    q_emb = embed_text(query)
    
    try:
        # Query Pinecone
        results = pinecone_index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        matches = []
        if isinstance(results, dict):
            matches = results.get('matches', [])
        else:
            matches = getattr(results, 'matches', []) or []
        for match in matches:
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            
            if text:
                chunks.append({
                    "document": text,
                    "metadata": metadata
                })
        
        return chunks
    except Exception as e:
        print(f"⚠️ Pinecone query error: {e}")
        return []

# === External Search ===

def external_search(query: str, max_results: int = 3) -> List[Dict]:
    if not SERPAPI_API_KEY:
        print("ℹ️ SERPAPI_API_KEY not set; skipping external web search.")
        return []

    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_API_KEY,
        "num": max_results,
        "hl": "en",
        "gl": "us",
    }
    try:
        result = GoogleSearch(params).get_dict()
    except Exception as e:
        print(f"⚠️ SerpAPI error: {e}")
        return []

    external = []
    for r in result.get("organic_results", [])[:max_results]:
        url = r.get("link")
        title = r.get("title") or "Untitled"
        snippet = r.get("snippet") or ""
        if url and "fcc.gov" not in url.lower():
            external.append({"title": title, "url": url, "content": snippet})
    return external

def fetch_full_text(url: str) -> str:
    """Fetch best-effort readable text from a URL.
    Uses a realistic User-Agent and attempts multiple selectors before falling back to <p> tags.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try common article/content containers first
        candidates = []
        for selector in [
            "article",
            "div.article",
            "div.post",
            "div#content",
            "main",
            "div.content",
        ]:
            node = soup.select_one(selector)
            if node:
                text = "\n".join(p.get_text().strip() for p in node.find_all(["p", "li"]))
                if len(text) >= 150:
                    candidates.append(text)

        if candidates:
            # Pick the longest candidate
            best = max(candidates, key=len)
            return best

        # Fallback: all paragraphs on the page
        fallback = "\n".join(p.get_text().strip() for p in soup.find_all("p"))
        return fallback
    except Exception:
        return ""

def generate_doc_id(url: str, chunk_index: int, chunk_text: str | None = None) -> str:
    """Generate an ID for a document chunk.
    Strategies:
      - url (default): stable per URL and chunk index -> overwrites on repeat runs
      - content: based on the chunk content -> treats new content as a new vector
    """
    try:
        if ID_STRATEGY.lower() == "content" and chunk_text:
            ch = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            return f"webc_{ch}"
        # Fallback/default: URL-based stable IDs
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"web_{url_hash}_chunk_{chunk_index:03d}"
    except Exception:
        # Last resort: random-like but deterministic per input
        base = (url or "") + "|" + str(chunk_index) + "|" + (chunk_text or "")
        h = hashlib.md5(base.encode()).hexdigest()[:12]
        return f"web_{h}"

def save_external_docs_to_pinecone(external_docs: List[Dict]) -> int:
    """Save external docs to Pinecone as chunks with embeddings.
    Returns the count of vectors that were actually upserted (new or updated) according to Pinecone response.
    """
    vectors_to_upsert = []

    for d in external_docs:
        url = d.get("url", "")
        title = d.get("title", "External Source")
        content = d.get("content", "")

        if not url:
            # Skip if no URL (shouldn't happen)
            continue

        # If we failed to fetch full text and only have a short snippet, still allow if it's reasonably informative
        if not content or len(content.strip()) < MIN_ARTICLE_LENGTH:
            # Try to enrich minimal content with title context before skipping
            combined = (title + "\n\n" + (content or "")).strip()
            if len(combined) >= MIN_ARTICLE_LENGTH:
                content = combined
            else:
                # Debug hint in CLI
                print(f"ℹ️ Skipping (short/empty): {url}")
                continue

        chunks = chunk_text(content)
        embeddings = embed_texts(chunks)

        today = str(datetime.date.today())
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = generate_doc_id(url, idx, chunk)
            vector = {
                'id': doc_id,
                'values': emb,
                'metadata': {
                    'text': chunk[:1000],  # Limit metadata size for Pinecone
                    'source': url,
                    'title': title,
                    'retrieved': today,
                    'chunk_index': idx,
                }
            }
            vectors_to_upsert.append(vector)

    if vectors_to_upsert:
        try:
            # Upload in batches of 100 and sum upserted counts
            batch_size = 100
            upserted_total = 0
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                resp = pinecone_index.upsert(vectors=batch)
                # Support both dict-like and object-like responses
                if isinstance(resp, dict):
                    upserted_total += int(resp.get('upserted_count', 0))
                else:
                    upserted_total += int(getattr(resp, 'upserted_count', 0))
            return upserted_total
        except Exception as e:
            print(f"⚠️ Failed saving external docs to Pinecone: {e}")
            return 0
    return 0

def save_external_docs_to_chroma(external_docs: List[Dict]) -> None:
    """Save external docs (with full text content) into ChromaDB as chunks with embeddings."""
    batched_ids: List[str] = []
    batched_docs: List[str] = []
    batched_embs: List[List[float]] = []
    batched_meta: List[Dict] = []

    for d in external_docs:
        url = d.get("url", "")
        title = d.get("title", "External Source")
        content = d.get("content", "")
        if not url or not content or len(content) < MIN_ARTICLE_LENGTH:
            continue

        chunks = chunk_text(content)
        embeddings = embed_texts(chunks)

        today = str(datetime.date.today())
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            batched_ids.append(str(uuid4()))
            batched_docs.append(chunk)
            batched_embs.append(emb)
            batched_meta.append({
                "source": url,
                "title": title,
                "retrieved": today,
                "chunk_index": idx,
            })

    if batched_ids:
        try:
            collection.add(
                ids=batched_ids,
                documents=batched_docs,
                embeddings=batched_embs,
                metadatas=batched_meta,
            )
        except Exception as e:
            # Non-fatal: continue chat even if persistence fails
            print(f"⚠️ Failed saving external docs to ChromaDB: {e}")

# === Prompt Construction ===

def build_prompt(query: str,
                 embedded_chunks: List[Dict],
                 external_docs: List[Dict]) -> str:
    system_instructions = (
        "You are an expert on emergency alert systems (EAS, WEA, IPAWS), public safety communications, and regulatory frameworks. "
        "Provide detailed, specific answers using the context below.\n\n"
        "Guidelines:\n"
        "- Include specific details: dates, names, statistics, and technical terms (EAS, WEA, IPAWS, CAP, FCC Part 11)\n"
        "- Cite sources using the format: 'According to [document/source]...'\n"
        "- Provide examples and context when helpful\n"
        "- List all sources at the end under '📚 Sources:' with markdown links\n"
        
    )
    #- If context is insufficient, supplement with your knowledge but indicate this clearly"
    parts = []

    for chunk in embedded_chunks:
        meta = chunk["metadata"]
        title = meta.get("title", "Embedded Document")
        url = meta.get("source") or meta.get("url", "")
        parts.append(f"Title: {title}" + (f" (URL: {url})" if url else "") + f"\n{chunk['document']}")

    for d in external_docs:
        title = d.get("title", "External Source")
        url = d.get("url", "")
        parts.append(f"Title: {title}" + (f" (URL: {url})" if url else "") + f"\n{d.get('content', '')}")

    context_text = "\n---\n".join(parts)

    return (
        f"{system_instructions}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def parse_sources(answer: str) -> Tuple[str, List[Tuple[str, str]]]:
    marker = "\n📚 Sources:"
    if marker in answer:
        ans_part, src_part = answer.split(marker, 1)
        sources = []
        for line in src_part.strip().splitlines():
            if line.startswith("- [") and "](" in line:
                try:
                    title = line.split("[", 1)[1].split("]")[0]
                    url = line.split("(", 1)[1].split(")")[0]
                    sources.append((title, url))
                except Exception:
                    continue
        return ans_part.strip(), sources
    return answer.strip(), []

# === Chat Loop ===

def chat():
    print("Chat Assistant - Powered by Pinecone (type 'exit' or Ctrl-C to quit)")
    try:
        stats = pinecone_index.describe_index_stats()
        initial_count = stats.total_vector_count
        print(f"🔢 Pinecone total embeddings at start: {initial_count}")
    except Exception as e:
        print(f"⚠️ Could not retrieve initial Pinecone count: {e}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Check if question is relevant to emergency systems using cosine similarity
            is_relevant, similarity_score = is_relevant_to_emergency_systems(user_input)
            
            if not is_relevant:
                print(f"\n🚫 I can only assist with questions related to emergency alert systems, "
                      "public safety communications, disaster response, cybersecurity policy, "
                      "and related regulatory topics. Please ask a question within my area of expertise.")
                continue

            embedded_chunks = retrieve_relevant_chunks(user_input)
            external_docs = external_search(user_input)

            for d in external_docs:
                full = fetch_full_text(d["url"])
                if full:
                    d["content"] = full

            # Persist the fetched external sources into Pinecone
            try:
                import time as time_module
                before_stats = pinecone_index.describe_index_stats()
                before_cnt = before_stats.total_vector_count
                
                _ = save_external_docs_to_pinecone(external_docs)
                
                # Brief pause for index stats to catch up
                time_module.sleep(2)
                after_stats = pinecone_index.describe_index_stats()
                after_cnt = after_stats.total_vector_count
                new_added = 0
                if before_cnt is not None and after_cnt is not None:
                    new_added = max(after_cnt - before_cnt, 0)
                print(f"✅ Added {new_added} new embeddings. Pinecone total: {after_cnt}")
            except Exception as e:
                print(f"⚠️ Error while saving external sources to Pinecone: {e}")

            if not embedded_chunks and not external_docs:
                print(f"Assistant: {FALLBACK_TEXT}")
                continue

            prompt = build_prompt(user_input, embedded_chunks, external_docs)

            response = None
            for attempt in range(3):
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=MAX_RESPONSE_TOKENS,
                        temperature=0.3,
                    )
                    break
                except Exception as e:
                    print(f"API error (attempt {attempt+1}): {e}")
                    time.sleep(1)

            if not response:
                print("Assistant: Sorry, I couldn't get a response.")
                continue

            full_answer = response.choices[0].message.content.strip()
            ans_text, sources = parse_sources(full_answer)

            print(f"\nAssistant: {ans_text}\n")
            print("📚 Sources:" if sources else "📚 Sources: None cited.")
            for title, url in sources:
                print(f"- [{title}]({url})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    chat()