"""
Fix Pinecone index dimension
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fcc-chatbot-index")

print("🔧 Fixing Pinecone Index Dimension\n")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete old index
print(f"🗑️  Deleting index '{PINECONE_INDEX}' with wrong dimension...")
try:
    pc.delete_index(PINECONE_INDEX)
    print("✅ Deleted\n")
except Exception as e:
    print(f"⚠️  {e}\n")

# Wait a moment for deletion
import time
time.sleep(2)

# Create new index with correct dimension
print("📝 Creating new index with dimension 1536 (text-embedding-ada-002)...")
pc.create_index(
    name=PINECONE_INDEX,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
print("✅ Index created with dimension 1536\n")

print("🎉 Ready! Run simple_migrate.py again")
