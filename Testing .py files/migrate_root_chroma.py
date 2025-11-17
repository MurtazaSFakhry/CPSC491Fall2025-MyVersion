#!/usr/bin/env python3
"""
Migrate ALL 5,309 embeddings from root ChromaDB to Pinecone
"""
import chromadb
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'fcc-chatbot-index')

print("🔧 Root ChromaDB → Pinecone Migration")
print(f"  ChromaDB: ./chroma_fcc_storage")
print(f"  Pinecone Index: {PINECONE_INDEX}\n")

# Connect to root ChromaDB
print("📂 Connecting to root ChromaDB...")
chroma_client = chromadb.PersistentClient(path='./chroma_fcc_storage')
collection = chroma_client.get_or_create_collection('fcc_documents')
print(f"✅ Found {collection.count()} embeddings\n")

# Connect to Pinecone
print("🌲 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
current_stats = index.describe_index_stats()
print(f"📊 Current Pinecone vectors: {current_stats.total_vector_count}\n")

# Confirm migration
response = input(f"Upload {collection.count()} embeddings? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("❌ Migration cancelled")
    exit(0)

print("\n🚀 Migrating...")

# Retrieve all records WITH embeddings
try:
    print("📥 Fetching all records with embeddings...")
    results = collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )
    
    if not results['embeddings']:
        print("❌ ERROR: No embeddings found in ChromaDB!")
        print("   The embeddings might not have been stored in this collection.")
        exit(1)
    
    print(f"✅ Retrieved {len(results['ids'])} records with embeddings")
    
    # Check embedding dimension
    if results['embeddings'] and len(results['embeddings']) > 0:
        dim = len(results['embeddings'][0])
        print(f"📏 Embedding dimension: {dim}")
        
        if dim != 1536:
            print(f"⚠️  WARNING: Expected 1536 dimensions, got {dim}")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Migration cancelled")
                exit(0)
    
    # Prepare vectors for Pinecone
    vectors = []
    for i, (doc_id, embedding, metadata, document) in enumerate(zip(
        results['ids'], 
        results['embeddings'], 
        results['metadatas'],
        results['documents']
    )):
        # Add document text to metadata
        meta = metadata.copy() if metadata else {}
        meta['text'] = document if document else ''
        
        vectors.append({
            'id': doc_id,
            'values': embedding,
            'metadata': meta
        })
    
    print(f"✅ Prepared {len(vectors)} vectors\n")
    
    # Upload in batches
    batch_size = 100
    total_uploaded = 0
    
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading"):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        total_uploaded += len(batch)
    
    print(f"\n✅ Migration complete!")
    print(f"   Uploaded: {total_uploaded} vectors\n")
    
    # Verify
    import time
    time.sleep(2)  # Wait for index to update
    final_stats = index.describe_index_stats()
    print(f"📊 Pinecone now has: {final_stats.total_vector_count} vectors\n")
    print("🎉 Done! Set USE_PINECONE=true in your .env")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
