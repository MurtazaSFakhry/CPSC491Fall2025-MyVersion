# Pinecone Integration Guide

## Overview
Your FCC Emergency Alert chatbot now uses **Pinecone** cloud vector storage for persistent, scalable embeddings storage.

## Architecture

```
User Query
    ↓
ChromaChat2.py
    ↓
1. Query Pinecone (1,096 vectors)
    ↓
2. Good match found? → Generate answer with sources
    ↓
3. No good match? → Search web with SerpAPI
    ↓
4. Process web results → Generate embeddings
    ↓
5. Save to Pinecone for future queries
    ↓
6. Generate answer with sources
```

## Configuration

### Environment Variables (.env)
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_key

# SerpAPI for web search fallback
SERPAPI_API_KEY=your_serpapi_key

# Pinecone cloud storage
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=fcc-chatbot-index
USE_PINECONE=true
```

## Current Status

### Pinecone Index
- **Name**: `fcc-chatbot-index`
- **Dimension**: 1536 (text-embedding-3-small with 1536 dims)
- **Metric**: Cosine similarity
- **Total Vectors**: 1,096
- **Cloud**: AWS (us-east-1)

### Data Sources
1. **VectordB ChromaDB**: 47 embeddings (migrated)
2. **doc/text-files-2025**: 10 text files → 1,049 embeddings

### Missing Data
- **Root ChromaDB**: 5,309 embeddings (CORRUPTED - cannot recover)
- **doc/text-files**: 22 text files (not yet processed)

## Usage

### Interactive Chat
```bash
cd VectordB
python -c "import ChromaChat2; ChromaChat2.chat()"
```

**Features:**
- Retrieves from Pinecone (1,096 vectors)
- Falls back to web search if no good match
- Automatically saves new findings to Pinecone
- Shows source citations

### Example Session
```
You: What are wireless emergency alerts?

[System searches Pinecone]
[Finds relevant chunks]
[Generates answer with sources]