"""
Migrate ChromaDB embeddings to Pinecone

This script reads all embeddings from your local ChromaDB and uploads them to Pinecone.
Run this once to populate your Pinecone index.

Usage:
    python VectordB/migrate_chroma_to_pinecone.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_fcc_storage")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "fcc_documents")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fcc-chatbot-index")

print("🔧 Migration Configuration:")
print(f"  ChromaDB Path: {CHROMA_PATH}")
print(f"  ChromaDB Collection: {CHROMA_COLLECTION}")
print(f"  Pinecone Index: {PINECONE_INDEX}")
print()

# Validate API key
if not PINECONE_API_KEY:
    print("❌ Error: PINECONE_API_KEY not found in .env file")
    print("   Please add: PINECONE_API_KEY=your-key-here")
    sys.exit(1)

# Connect to ChromaDB
print("📂 Connecting to ChromaDB...")
try:
    chroma_client = PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
    total_count = collection.count()
    print(f"✅ Found {total_count:,} embeddings in ChromaDB")
except Exception as e:
    print(f"❌ Error connecting to ChromaDB: {e}")
    sys.exit(1)

# Connect to Pinecone
print("\n🌲 Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes] if hasattr(existing_indexes, '__iter__') else []
    
    if PINECONE_INDEX not in index_names:
        print(f"📝 Creating new index: {PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,  # text-embedding-3-small dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')  # adjust region as needed
        )
        print("✅ Index created")
    else:
        print(f"✅ Index '{PINECONE_INDEX}' already exists")
    
    index = pc.Index(PINECONE_INDEX)
    
    # Check current stats
    stats = index.describe_index_stats()
    current_count = stats.get('total_vector_count', 0) if isinstance(stats, dict) else getattr(stats, 'total_vector_count', 0)
    print(f"📊 Current vectors in Pinecone: {current_count:,}")
    
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    sys.exit(1)

# Ask for confirmation
print(f"\n⚠️  This will upload {total_count:,} embeddings to Pinecone index '{PINECONE_INDEX}'")
if current_count > 0:
    print(f"   Note: Index already has {current_count:,} vectors")
response = input("   Continue? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("❌ Migration cancelled")
    sys.exit(0)

# Migrate data
print("\n🚀 Starting migration...")
BATCH_SIZE = 100

# Fetch data using peek to get all records
print("📥 Fetching data from ChromaDB...")
try:
    # Use peek to get a large sample (ChromaDB's way of getting all data)
    all_data = collection.peek(limit=total_count)
    
    all_ids = all_data['ids']
    all_embeddings = all_data['embeddings']
    all_documents = all_data['documents']
    all_metadatas = all_data['metadatas']
    
    print(f"✅ Retrieved {len(all_ids):,} records")
    
except Exception as e:
    print(f"❌ Error fetching from ChromaDB: {e}")
    print("\n⚠️  ChromaDB may have internal issues. Trying alternative export method...")
    
    # Alternative: Export to JSONL first
    print("📝 Exporting ChromaDB to JSONL format...")
    try:
        import json
        from uuid import uuid4
        
        # Query in smaller chunks and export
        export_file = "chroma_export_temp.jsonl"
        with open(export_file, "w", encoding="utf-8") as f:
            batch_size = 100
            for i in tqdm(range(0, total_count, batch_size)):
                # Query a batch
                results = collection.query(
                    query_embeddings=[[0.0] * 1536],  # dummy query with correct dimension
                    n_results=min(batch_size, total_count - i),
                    include=["embeddings", "documents", "metadatas"]
                )
                
                for j in range(len(results['ids'][0])):
                    record = {
                        'id': results['ids'][0][j],
                        'embedding': results['embeddings'][0][j],
                        'document': results['documents'][0][j],
                        'metadata': results['metadatas'][0][j]
                    }
                    f.write(json.dumps(record) + "\n")
        
        print(f"✅ Exported to {export_file}")
        print("❌ Please use upload_to_pinecone.py with this JSONL file instead")
        sys.exit(1)
        
    except Exception as e2:
        print(f"❌ Alternative export also failed: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Upload to Pinecone in batches
print(f"\n📤 Uploading to Pinecone (batch size: {BATCH_SIZE})...")
uploaded = 0
errors = 0

for i in tqdm(range(0, len(all_ids), BATCH_SIZE)):
    batch_end = min(i + BATCH_SIZE, len(all_ids))
    batch_ids = all_ids[i:batch_end]
    batch_embeddings = all_embeddings[i:batch_end]
    batch_documents = all_documents[i:batch_end]
    batch_metadatas = all_metadatas[i:batch_end]
    
    # Prepare vectors for Pinecone
    vectors = []
    for j in range(len(batch_ids)):
        metadata = batch_metadatas[j] if batch_metadatas[j] else {}
        # Add the document text to metadata so it can be retrieved
        metadata['text'] = batch_documents[j]
        
        vectors.append({
            'id': batch_ids[j],
            'values': batch_embeddings[j],
            'metadata': metadata
        })
    
    # Upload batch
    try:
        index.upsert(vectors=vectors)
        uploaded += len(vectors)
    except Exception as e:
        print(f"\n⚠️  Error uploading batch {i}-{batch_end}: {e}")
        errors += 1

print(f"\n✅ Migration complete!")
print(f"   Uploaded: {uploaded:,} vectors")
if errors > 0:
    print(f"   Errors: {errors} batches failed")

# Verify final count
print("\n🔍 Verifying migration...")
try:
    stats = index.describe_index_stats()
    final_count = stats.get('total_vector_count', 0) if isinstance(stats, dict) else getattr(stats, 'total_vector_count', 0)
    print(f"📊 Final count in Pinecone: {final_count:,}")
    
    if final_count >= total_count:
        print("✅ Migration successful! All embeddings uploaded.")
    else:
        print(f"⚠️  Warning: Expected {total_count:,} but found {final_count:,} in Pinecone")
except Exception as e:
    print(f"⚠️  Could not verify: {e}")

print("\n🎉 Done! Your Streamlit app can now use Pinecone.")
print("   Make sure USE_PINECONE=true in your .env or Streamlit secrets")
