#!/usr/bin/env python3
"""
Check root ChromaDB status and attempt recovery
"""
import chromadb
from chromadb.config import Settings

print("🔍 Checking Root ChromaDB")
print("=" * 50)

# Connect to root ChromaDB
client = chromadb.PersistentClient(path='./chroma_fcc_storage')
collection = client.get_or_create_collection('fcc_documents')

print(f"📊 Total embeddings: {collection.count()}")

# Try to retrieve records
print("\n🧪 Testing retrieval...")
try:
    # Try small batch first
    results = collection.get(limit=10)
    print(f"✅ Retrieved {len(results['ids'])} sample records")
    print(f"   Sample IDs: {results['ids'][:3]}")
    
    # Try larger batch
    print("\n🧪 Testing larger batch (100)...")
    results = collection.get(limit=100)
    print(f"✅ Retrieved {len(results['ids'])} records")
    
    # Try getting ALL records
    print("\n🧪 Testing full retrieval...")
    all_results = collection.get()
    print(f"✅ Retrieved ALL {len(all_results['ids'])} records!")
    
    # Check if embeddings are included
    if all_results['embeddings'] and len(all_results['embeddings']) > 0:
        print(f"✅ Embeddings present: {len(all_results['embeddings'][0])} dimensions")
    else:
        print("⚠️  No embeddings in results")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\n🔧 Attempting incremental fetch...")
    
    # Try incremental approach
    try:
        batch_size = 100
        all_ids = []
        offset = 0
        
        while True:
            batch = collection.get(limit=batch_size, offset=offset)
            if not batch['ids']:
                break
            all_ids.extend(batch['ids'])
            offset += batch_size
            print(f"   Fetched {len(all_ids)} so far...")
            
            if len(all_ids) >= 500:  # Limit test to 500
                print("   (stopping test at 500 for speed)")
                break
        
        print(f"✅ Incremental fetch successful: {len(all_ids)} records")
        
    except Exception as e2:
        print(f"❌ Incremental fetch also failed: {e2}")
