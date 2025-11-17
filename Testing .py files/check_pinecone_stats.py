#!/usr/bin/env python3
"""Quick script to check Pinecone index statistics"""
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'fcc-chatbot-index')

if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not set in .env")
    exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

print(f"🔍 Checking Pinecone Index: {PINECONE_INDEX}\n")

stats = index.describe_index_stats()

print(f"📊 Statistics:")
print(f"   Total vectors: {stats.total_vector_count}")
print(f"   Dimension: {stats.dimension}")
print(f"   Index fullness: {stats.index_fullness}")
print(f"   Namespaces: {stats.namespaces}")

# Calculate approximate storage
vectors = stats.total_vector_count
dimension = stats.dimension
bytes_per_float = 4
total_bytes = vectors * dimension * bytes_per_float
total_mb = total_bytes / (1024 * 1024)
total_gb = total_mb / 1024

print(f"\n💾 Estimated Storage:")
print(f"   Vectors × Dimensions × 4 bytes")
print(f"   {vectors:,} × {dimension} × 4 = {total_bytes:,} bytes")
print(f"   ≈ {total_mb:.2f} MB")
print(f"   ≈ {total_gb:.4f} GB")

print("\n✅ Done!")
