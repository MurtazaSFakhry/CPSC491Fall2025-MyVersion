"""
Simple Pinecone migration - handles ChromaDB corruption gracefully

This version queries ChromaDB one item at a time to avoid internal errors.
"""

import os
import sys
from dotenv import load_dotenv
from chromadb import PersistentClient
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fcc-chatbot-index")

# Try VectordB ChromaDB first (more reliable)
CHROMA_PATH = "./VectordB/chroma_fcc_storage"
CHROMA_COLLECTION = "fcc_documents"

print("🔧 Simple Migration Script")
print(f"  ChromaDB: {CHROMA_PATH}")
print(f"  Pinecone Index: {PINECONE_INDEX}\n")

if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not found")
    sys.exit(1)

# Connect to ChromaDB
print("📂 Connecting to ChromaDB...")
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
total = collection.count()
print(f"✅ Found {total} embeddings\n")

# Connect to Pinecone
print("🌲 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check/create index
indexes = pc.list_indexes()
index_names = [idx.name for idx in indexes] if hasattr(indexes, '__iter__') else []

if PINECONE_INDEX not in index_names:
    print(f"📝 Creating index with dimension 384 (text-embedding-3-small)...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,  # text-embedding-3-small
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print("✅ Index created")

index = pc.Index(PINECONE_INDEX)

# Check current count
try:
    stats = index.describe_index_stats()
    current = stats.get('total_vector_count', 0) if isinstance(stats, dict) else getattr(stats, 'total_vector_count', 0)
    print(f"📊 Current Pinecone vectors: {current}\n")
except:
    current = 0

response = input(f"Upload {total} embeddings? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    sys.exit(0)

print("\n🚀 Migrating...")

# Get all IDs first
try:
    all_data = collection.get(limit=total, include=["embeddings", "documents", "metadatas"])
    ids = all_data['ids']
    embeddings = all_data['embeddings']
    documents = all_data['documents']
    metadatas = all_data['metadatas']
except Exception as e:
    print(f"⚠️  Full fetch failed: {e}")
    print("Trying incremental fetch...")
    
    # Fallback: query in batches
    ids, embeddings, documents, metadatas = [], [], [], []
    batch_size = 10
    for i in range(0, total, batch_size):
        try:
            batch = collection.get(
                limit=min(batch_size, total - i),
                offset=i,
                include=["embeddings", "documents", "metadatas"]
            )
            ids.extend(batch['ids'])
            embeddings.extend(batch['embeddings'])
            documents.extend(batch['documents'])
            metadatas.extend(batch['metadatas'])
        except:
            print(f"⚠️  Skipped batch at {i}")
            continue

print(f"✅ Retrieved {len(ids)} records\n")

# Upload to Pinecone
batch_size = 100
uploaded = 0

for i in tqdm(range(0, len(ids), batch_size), desc="Uploading"):
    batch_end = min(i + batch_size, len(ids))
    
    vectors = []
    for j in range(i, batch_end):
        try:
            metadata = metadatas[j] if metadatas[j] else {}
            metadata['text'] = documents[j]
            
            vectors.append({
                'id': ids[j],
                'values': embeddings[j],
                'metadata': metadata
            })
        except Exception as e:
            print(f"\n⚠️  Skipped vector {j}: {e}")
            continue
    
    if vectors:
        try:
            index.upsert(vectors=vectors)
            uploaded += len(vectors)
        except Exception as e:
            print(f"\n⚠️  Batch upload failed: {e}")

print(f"\n✅ Migration complete!")
print(f"   Uploaded: {uploaded} vectors\n")

# Verify
try:
    stats = index.describe_index_stats()
    final = stats.get('total_vector_count', 0) if isinstance(stats, dict) else getattr(stats, 'total_vector_count', 0)
    print(f"📊 Pinecone now has: {final} vectors")
except:
    pass

print("\n🎉 Done! Set USE_PINECONE=true in your .env")
