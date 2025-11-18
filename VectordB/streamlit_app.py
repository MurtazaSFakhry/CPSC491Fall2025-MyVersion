"""
Streamlit UI for Emergency Alert Systems Chat Assistant

Usage:
  streamlit run VectordB/streamlit_app.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None
from typing import List, Dict

try:
    from serpapi import GoogleSearch
except ImportError:
    from serpapi.google_search import GoogleSearch

# Load environment variables
load_dotenv()

# === Configuration ===
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
FALLBACK_TEXT = "No information available in the dataset or external sources for that question."
#RELEVANCE_THRESHOLD = 0.35

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME") or "fcc-chatbot-index"
ID_STRATEGY = os.getenv("PINECONE_ID_STRATEGY", "url")  # 'url' (default) or 'content'

# Override with Streamlit secrets if available
try:
    if hasattr(st, 'secrets') and st.secrets:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
        SERPAPI_API_KEY = st.secrets.get("SERPAPI_KEY", st.secrets.get("SERPAPI_API_KEY", SERPAPI_API_KEY))
        PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", PINECONE_API_KEY)
        PINECONE_INDEX = st.secrets.get("PINECONE_INDEX", st.secrets.get("PINECONE_INDEX_NAME", PINECONE_INDEX))
except:
    pass

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone client (required)
pc = None
pinecone_index = None
if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not set. Please configure your Pinecone API key.")
    st.stop()
if Pinecone is None:
    st.error("❌ Pinecone library not installed. Run: pip install pinecone-client")
    st.stop()

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX)
except Exception as e:
    st.error(f"❌ Failed to initialize Pinecone: {e}")
    st.stop()

# Import helper functions from ChromaChat2
sys.path.insert(0, str(ROOT / "VectordB"))
from ChromaChat2 import (
    embed_text,
    external_search,
    fetch_full_text,
    build_prompt,
    parse_sources,
    is_relevant_to_emergency_systems,
    EMERGENCY_TOPICS
)

# === Helper Functions ===

MIN_ARTICLE_LENGTH = 200
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

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

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMENSIONS,
    )
    return [r.embedding for r in resp.data]

def generate_doc_id(url: str, chunk_index: int, chunk_text: str | None = None) -> str:
    """Generate an ID for a document chunk.
    Strategies:
      - url (default): stable per URL and chunk index -> overwrites on repeat runs
      - content: based on the chunk content -> treats new content as a new vector
    """
    import hashlib as _hashlib
    try:
        if ID_STRATEGY.lower() == "content" and chunk_text:
            ch = _hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            return f"webc_{ch}"
        # Fallback/default: URL-based stable IDs
        url_hash = _hashlib.md5(url.encode()).hexdigest()[:8]
        return f"web_{url_hash}_chunk_{chunk_index:03d}"
    except Exception:
        base = (url or "") + "|" + str(chunk_index) + "|" + (chunk_text or "")
        h = _hashlib.md5(base.encode()).hexdigest()[:12]
        return f"web_{h}"

def save_external_docs_to_pinecone(external_docs: List[Dict]) -> int:
    """Save external docs to Pinecone as chunks with embeddings.
    Returns the count of vectors that were actually upserted (new or updated) according to Pinecone response.
    """
    vectors_to_upsert = []

    import datetime as _dt
    today = str(_dt.date.today())

    for d in external_docs:
        url = d.get("url", "")
        title = d.get("title", "External Source")
        content = d.get("content", "")

        if not url:
            continue

        if not content or len(content.strip()) < MIN_ARTICLE_LENGTH:
            combined = (title + "\n\n" + (content or "")).strip()
            if len(combined) >= MIN_ARTICLE_LENGTH:
                content = combined
            else:
                # Hint in sidebar for transparency
                st.sidebar.info(f"Skipping short/empty: {url}")
                continue

        chunks = chunk_text(content)
        embeddings = embed_texts(chunks)

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = generate_doc_id(url, idx, chunk)
            vectors_to_upsert.append({
                'id': doc_id,
                'values': emb,
                'metadata': {
                    'text': chunk[:1000],
                    'source': url,
                    'title': title,
                    'retrieved': today,
                    'chunk_index': idx,
                }
            })

    if not vectors_to_upsert:
        return 0

    # Upsert in batches
    upserted_total = 0
    batch_size = 100
    try:
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            resp = pinecone_index.upsert(vectors=batch)
            if isinstance(resp, dict):
                upserted_total += int(resp.get('upserted_count', 0))
            else:
                upserted_total += int(getattr(resp, 'upserted_count', 0))
    except Exception as e:
        st.sidebar.warning(f"⚠️ Failed saving to Pinecone: {e}")
        return 0

    return upserted_total

def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K) -> List[Dict]:
    """Retrieve relevant chunks from Pinecone."""
    q_emb = embed_text(query)
    results = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = results.get("matches", []) if isinstance(results, dict) else results.matches
    chunks = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        text = md.get("text") or md.get("chunk") or md.get("document") or ""
        meta = {
            "title": md.get("title", "Pinecone Document"),
            "source": md.get("source") or md.get("url") or "",
        }
        if text:
            chunks.append({"document": text, "metadata": meta})
    return chunks

# === Streamlit Configuration ===
st.set_page_config(
    page_title="Emergency Alert Systems Chat",
    page_icon="🚨",
    layout="wide"
)

def main():
    """Main Streamlit UI"""
    st.title("🚨 Emergency Alert Systems Chat Assistant")
    st.markdown("Ask questions about EAS, WEA, IPAWS, and emergency communications!")
    
    # Sidebar with stats
    with st.sidebar:
        st.header("📊 System Stats")
        st.success(f"Using Pinecone index: {PINECONE_INDEX}")
        try:
            stats = pinecone_index.describe_index_stats()
            total_vecs = stats.get("total_vector_count") if isinstance(stats, dict) else getattr(stats, "total_vector_count", None)
            if total_vecs is not None:
                st.info(f"Vectors: {total_vecs:,}")
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve stats: {e}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask about EAS, WEA, IPAWS
        - Request specific regulations
        - Inquire about emergency procedures
        - Ask for FCC policy details
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about emergency alert systems..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching for relevant information..."):
                # Check relevance
                try:
                    is_relevant, similarity_score = is_relevant_to_emergency_systems(prompt)
                except Exception as e:
                    st.error(f"Error checking relevance: {e}")
                    is_relevant = True  # Default to allowing
                    similarity_score = 1.0
                
                if not is_relevant:
                    response_text = (
                        "🚫 I can only assist with questions related to emergency alert systems, "
                        "public safety communications, disaster response, cybersecurity policy, "
                        "and related regulatory topics. Please ask a question within my area of expertise."
                    )
                    st.warning(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    # Retrieve relevant chunks
                    try:
                        embedded_chunks = retrieve_relevant_chunks(prompt)
                        external_docs = external_search(prompt)
                        
                        # Fetch full text for external docs
                        for d in external_docs:
                            full = fetch_full_text(d["url"])
                            if full:
                                d["content"] = full
                        
                        # Save external docs to Pinecone
                        try:
                            before_stats = pinecone_index.describe_index_stats()
                            before_cnt = before_stats.get("total_vector_count") if isinstance(before_stats, dict) else getattr(before_stats, "total_vector_count", None)
                            _ = save_external_docs_to_pinecone(external_docs)
                            time.sleep(2)
                            after_stats = pinecone_index.describe_index_stats()
                            after_cnt = after_stats.get("total_vector_count") if isinstance(after_stats, dict) else getattr(after_stats, "total_vector_count", None)
                            new_added = 0
                            if before_cnt is not None and after_cnt is not None:
                                new_added = max(after_cnt - before_cnt, 0)
                            st.sidebar.success(f"✅ Added {new_added} new embeddings. Pinecone total: {after_cnt:,}")
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ Could not save external docs: {e}")
                        
                        if not embedded_chunks and not external_docs:
                            response_text = FALLBACK_TEXT
                            st.info(response_text)
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                        else:
                            # Build prompt and get response
                            prompt_text = build_prompt(prompt, embedded_chunks, external_docs)
                            
                            response = None
                            for attempt in range(3):
                                try:
                                    response = openai_client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[{"role": "system", "content": prompt_text}],
                                        max_tokens=MAX_RESPONSE_TOKENS,
                                        temperature=0.3,
                                    )
                                    break
                                except Exception as e:
                                    if attempt < 2:
                                        time.sleep(1)
                                    else:
                                        st.error(f"API error: {e}")
                            
                            if response:
                                full_answer = response.choices[0].message.content.strip()
                                ans_text, sources = parse_sources(full_answer)
                                
                                # Display answer
                                st.markdown(ans_text)
                                
                                # Display sources
                                if sources:
                                    st.markdown("\n📚 **Sources:**")
                                    for title, url in sources:
                                        st.markdown(f"- [{title}]({url})")
                                else:
                                    st.markdown("\n📚 **Sources:** None cited.")
                                
                                # Add to chat history
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": full_answer
                                })
                            else:
                                st.error("Sorry, I couldn't get a response. Please try again.")
                                
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        import traceback
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
