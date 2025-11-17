#!/usr/bin/env python3
"""
Process text files, generate embeddings, and upload to Pinecone
"""
import os
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
import time
import hashlib

load_dotenv()

# Configuration
TEXT_DIR = Path("doc/text-files-2025")
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks
BATCH_SIZE = 100  # vectors per batch upload
EMBEDDING_MODEL = "text-embedding-3-small"  # Using 1536 dimensions
EMBEDDING_DIMENSIONS = 1536  # Match Pinecone index dimension

# Initialize clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX', 'fcc-chatbot-index'))

print("🚀 Text Files → Pinecone Pipeline")
print("=" * 60)
print(f"📂 Source: {TEXT_DIR}")
print(f"🧠 Model: {EMBEDDING_MODEL}")
print(f"📏 Chunk size: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
print(f"🌲 Pinecone index: {os.getenv('PINECONE_INDEX')}\n")

# Check current Pinecone stats
current_stats = index.describe_index_stats()
print(f"📊 Current vectors in Pinecone: {current_stats.total_vector_count}\n")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_len:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only break if not too short
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def generate_embedding(text):
    """Generate embedding for text using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS  # Set to 1536 for Pinecone compatibility
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def generate_doc_id(filename, chunk_index):
    """Generate unique ID for document chunk"""
    # Create hash of filename for shorter IDs
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:6]
    return f"doc_{file_hash}_chunk_{chunk_index:03d}"

# Get all text files
text_files = sorted(TEXT_DIR.glob("*.txt"))
print(f"📁 Found {len(text_files)} text files\n")

if not text_files:
    print("❌ No text files found!")
    exit(1)

# Confirm processing
print("Files to process:")
for f in text_files:
    size_kb = f.stat().st_size / 1024
    print(f"  • {f.name} ({size_kb:.1f} KB)")

response = input(f"\n🔄 Process {len(text_files)} files and upload to Pinecone? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("❌ Processing cancelled")
    exit(0)

print("\n" + "=" * 60)
print("🔄 Processing files...\n")

all_vectors = []
total_chunks = 0

# Process each file
for file_path in text_files:
    print(f"📄 Processing: {file_path.name}")
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        continue
    
    # Chunk the content
    chunks = chunk_text(content)
    print(f"  ✂️  Split into {len(chunks)} chunks")
    
    # Generate embeddings for each chunk
    for i, chunk in enumerate(tqdm(chunks, desc=f"  Embedding", leave=False)):
        if not chunk.strip():
            continue
            
        # Generate embedding
        embedding = generate_embedding(chunk)
        if embedding is None:
            continue
        
        # Create vector
        doc_id = generate_doc_id(file_path.name, i)
        vector = {
            'id': doc_id,
            'values': embedding,
            'metadata': {
                'text': chunk[:1000],  # Limit metadata size
                'filename': file_path.name,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
        }
        all_vectors.append(vector)
        total_chunks += 1
        
        # Rate limit: OpenAI allows 3000 RPM for tier 1
        time.sleep(0.02)  # ~50 requests per second
    
    print(f"  ✅ Created {len(chunks)} embeddings\n")

print("=" * 60)
print(f"📦 Total vectors prepared: {len(all_vectors)}\n")

# Upload to Pinecone in batches
if all_vectors:
    print("☁️  Uploading to Pinecone...\n")
    uploaded = 0
    
    for i in tqdm(range(0, len(all_vectors), BATCH_SIZE), desc="Uploading"):
        batch = all_vectors[i:i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            uploaded += len(batch)
        except Exception as e:
            print(f"\n❌ Upload error for batch {i//BATCH_SIZE}: {e}")
            continue
    
    print(f"\n✅ Upload complete!")
    print(f"   Uploaded: {uploaded} vectors")
    
    # Verify final count
    time.sleep(2)
    final_stats = index.describe_index_stats()
    print(f"\n📊 Pinecone now has: {final_stats.total_vector_count} vectors")
    print(f"   Net increase: +{final_stats.total_vector_count - current_stats.total_vector_count}")
    
    print("\n🎉 Done! Your Pinecone index is ready to use.")
else:
    print("❌ No vectors created!")
