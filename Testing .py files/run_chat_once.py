"""
Utility to run a single ChromaChat2-like cycle non-interactively:
- reads query from argv
- retrieves Pinecone context
- does external search (if SERPAPI available) and fetches full text
- saves external docs to Pinecone
- prints Pinecone total with the number of new embeddings added

Usage (from repo root or VectordB/):
  python VectordB/run_chat_once.py "What are Wireless Emergency Alerts?"
"""
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Ensure we can import ChromaChat2 from this folder
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ChromaChat2 as CC


def main():
    if len(sys.argv) < 2:
        print("Provide a query, e.g.: python run_chat_once.py 'What is IPAWS?'")
        sys.exit(1)

    query = sys.argv[1]
    print(f"🔎 Query: {query}")

    # Stats before
    try:
        before = CC.pinecone_index.describe_index_stats()
        before_cnt = before.total_vector_count
    except Exception as e:
        print(f"⚠️ Could not fetch Pinecone stats before: {e}")
        before_cnt = None

    # Retrieval (not strictly needed but mirrors flow)
    _ = CC.retrieve_relevant_chunks(query)

    # External search and full text fetch
    external_docs = CC.external_search(query)
    for d in external_docs:
        full = CC.fetch_full_text(d.get("url", ""))
        if full:
            d["content"] = full

    # Save to Pinecone
    _ = CC.save_external_docs_to_pinecone(external_docs)

    # Stats after (with small wait)
    import time
    time.sleep(2)
    try:
        after = CC.pinecone_index.describe_index_stats()
        after_cnt = after.total_vector_count
        new_added = 0
        if before_cnt is not None and after_cnt is not None:
            new_added = max(after_cnt - before_cnt, 0)
        print(f"✅ Added {new_added} new embeddings. Pinecone total: {after_cnt}")
    except Exception as e:
        print(f"⚠️ Could not fetch Pinecone stats after: {e}")


if __name__ == "__main__":
    main()
