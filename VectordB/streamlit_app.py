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
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv

try:
    from serpapi import GoogleSearch
except ImportError:
    from serpapi.google_search import GoogleSearch

# Load environment variables
load_dotenv()

# === Configuration - Use correct paths ===
# Point to the root-level chroma_fcc_storage
PERSIST_PATH = str(ROOT / "chroma_fcc_storage")
COLLECTION_NAME = "fcc_documents"
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_TOP_K = 5
MAX_RESPONSE_TOKENS = 500
FALLBACK_TEXT = "No information available in the dataset or external sources for that question."
RELEVANCE_THRESHOLD = 0.35

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")

# Override with Streamlit secrets if available
try:
    if hasattr(st, 'secrets') and st.secrets:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
        SERPAPI_API_KEY = st.secrets.get("SERPAPI_KEY", st.secrets.get("SERPAPI_API_KEY", SERPAPI_API_KEY))
except:
    pass

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = PersistentClient(path=PERSIST_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

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

def retrieve_relevant_chunks(query: str, top_k: int = SIMILARITY_TOP_K):
    """Retrieve relevant chunks from ChromaDB"""
    q_emb = embed_text(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [{"document": doc, "metadata": meta} for doc, meta in zip(docs, metas)]

def save_external_docs_to_chroma(external_docs):
    """Save external docs to ChromaDB"""
    from uuid import uuid4
    import datetime
    
    batched_ids = []
    batched_docs = []
    batched_embs = []
    batched_meta = []
    
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    MIN_ARTICLE_LENGTH = 300
    
    def chunk_text(text: str):
        if len(text) <= CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
        return chunks
    
    def embed_texts(texts):
        if not texts:
            return []
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [r.embedding for r in resp.data]
    
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
            st.sidebar.warning(f"⚠️ Failed saving external docs: {e}")

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
        try:
            initial_count = collection.count()
            st.success(f"**ChromaDB Embeddings:** {initial_count:,}")
            st.info(f"**Database Path:** `{PERSIST_PATH}`")
        except Exception as e:
            st.error(f"⚠️ ChromaDB Error: {e}")
        
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
                        
                        # Save external docs to ChromaDB
                        try:
                            before_cnt = collection.count()
                            save_external_docs_to_chroma(external_docs)
                            after_cnt = collection.count()
                            added = after_cnt - before_cnt
                            if added > 0:
                                st.sidebar.success(f"✅ Added {added} embeddings. Total: {after_cnt:,}")
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
