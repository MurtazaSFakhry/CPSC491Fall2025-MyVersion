# FCC Emergency Alert Chatbot & Knowledge Retrieval Platform

> Intelligent Q&A, retrieval-augmented generation (RAG), and fine‑tuning pipeline for public safety / emergency alert system content.

## 1. Overview

This repository implements a full lifecycle for an emergency alert domain chatbot:

1. Source acquisition (PDFs, web pages, curated text files)
2. Preprocessing & normalization into structured JSONL training/eval corpora
3. Dataset validation & repair utilities
4. Optional OpenAI fine‑tuning pipeline
5. Embedding generation and migration from local ChromaDB to Pinecone serverless vector storage
6. Retrieval‑augmented chat (CLI and Streamlit UI) with automatic external web enrichment

The current production vector backend is **Pinecone** (`fcc-chatbot-index`) using `text-embedding-3-small` (1536 dims). Local **ChromaDB** is retained for troubleshooting and legacy migration scripts but excluded from version control to avoid large binary commits.

## 2. Key Features

- Hybrid retrieval: internal curated corpus + targeted external search (SerpAPI) with on-the-fly scraping
- Automatic embedding of newly discovered pages (length threshold & de-duplication strategies)
- Deterministic or growth-focused ID strategies for Pinecone (`url` vs `content`)
- Streamlit web UI (`VectordB/streamlit_app.py`) & CLI chat (`VectordB/ChromaChat2.py`)
- Dataset creation / merging / validation utilities (`archive/preprocessing/*`)
- Migration scripts: Chroma → Pinecone, dimensional fixes, bulk embedding upload
- Transparent logging: “Added X new embeddings. Pinecone total: Y”
- Domain filtering: excludes `fcc.gov` domains where required

## 3. High-Level Architecture

```
	       ┌──────────────────────────────┐
	       │     Raw Source Material      │
	       │  PDFs | Web Pages | Texts    │
	       └──────────────┬──────────────┘
			      │
		      (Ingestion / Scraping)
			      │
	       ┌──────────────▼──────────────┐
	       │  Preprocessing / Cleaning   │
	       │  create_jsonl, merge_jsonl  │
	       └──────────────┬──────────────┘
			      │
			Validation & Fixes
		     (format_validation, etc.)
			      │
		     Fine-Tuning (Optional)
		       train_gpt.py, chat.py
			      │
	       ┌──────────────▼──────────────┐
	       │  Embeddings Generation      │
	       │  text-embedding-3-small     │
	       └──────────────┬──────────────┘
			      │
		      Vector Storage Layer
		   Pinecone  <—>  ChromaDB* (legacy)
			      │
	       ┌──────────────▼──────────────┐
	       │     Retrieval & RAG Chat    │
	       │  Streamlit | CLI | External │
	       └──────────────────────────────┘
```
*ChromaDB retained mainly for migration & fallback, not primary.

## 4. Repository Structure (Selected)

| Path | Purpose |
|------|---------|
| `VectordB/ChromaChat2.py` | Core CLI chat with Pinecone retrieval & auto-save of external docs |
| `VectordB/streamlit_app.py` | Streamlit UI providing interactive chat and embedding persistence |
| `VectordB/run_chat_once.py` | Single-shot query test harness (good for verifying embeddings added) |
| `VectordB/PINECONE_INTEGRATION.md` | Notes on vector migration & Pinecone configuration |
| `VectordB/migrate_chroma_to_pinecone.py` | Bulk migration from local Chroma collections to Pinecone |
| `VectordB/simple_migrate.py` | Minimal migration example / quick start script |
| `VectordB/fix_pinecone_dimension.py` | Utilities when dimension mismatches occur |
| `VectordB/embed_and_upload_texts.py` | Batch embed raw text files & upload to Pinecone |
| `VectordB/check_root_chroma.py` | Inspect legacy Chroma root store |
| `VectordB/pinecone_chat.py` | Earlier Pinecone chat prototype (superseded by ChromaChat2) |
| `archive/preprocessing/*.py` | Dataset generation, merging, validation scripts |
| `archive/finetuning/*.py` | Fine-tuning orchestration / monitoring |
| `doc/datasets/*.jsonl` | Generated & validated datasets (intermediate & final forms) |
| `check_pinecone_stats.py` | Pinecone index stats utility (vector count, dimension) |
| `fixing-jsonl-files.py` | Repairs / improvements for existing JSONL training data |

## 5. Data Pipeline Details

### 5.1 Ingestion
- PDF and web scraping utilities gather raw domain-relevant content.
- Pages shorter than a configurable minimum (default ~200 chars after cleaning) are skipped.

### 5.2 Preprocessing
Scripts in `archive/preprocessing/` transform raw text and PDF extracts into structured JSONL with fields like `prompt`, `completion`, metadata tags, and optionally source attribution.

### 5.3 Validation & Repair
`format_validation.py`, `validate_dataset.py`, and `fixing-jsonl-files.py` check for:
- JSONL structural errors
- Missing fields / empty completions
- Length constraints for prompts/completions
- Duplicate or near-duplicate entries

### 5.4 Fine-Tuning (Optional)
Located in `archive/finetuning/` (e.g., `train_gpt.py`, `individual_finetune_chat.py`). These scripts assume properly formatted JSONL files meeting provider requirements. They may require mapping into OpenAI’s fine-tuning schema or similar service-specific formatting.

### 5.5 Embedding Generation & Storage
- Current embedding model: `text-embedding-3-small` (dimension 1536)
- Migration scripts moved legacy Chroma vectors into Pinecone serverless index `fcc-chatbot-index`.
- New content discovered via queries is chunked and embedded on-the-fly.

### 5.6 Retrieval & RAG
- Combines internal Pinecone semantic search with targeted SerpAPI queries when context insufficient.
- External results (HTML pages) scraped, cleaned, chunked, embedded, then upserted back into Pinecone (growth loop).

## 6. Environment & Dependencies

### 6.1 Python Version
Developed & tested on Python 3.13 (earlier 3.11+ likely compatible). Use virtual environments for isolation.

### 6.2 Installation (All-In-One)
```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
For minimal chat-only environment you can alternatively install `VectordB/requirements.txt`.

### 6.3 Core Libraries
- `openai` – embeddings & (optionally) chat/fine-tuning
- `pinecone-client` – vector index operations
- `chromadb` – legacy / fallback local store
- `google-search-results` (SerpAPI) – targeted web search
- `beautifulsoup4`, `requests` – scraping / HTML parsing
- `streamlit` – web UI
- `tqdm`, `numpy`, `scikit-learn` – progress, array ops, auxiliary tooling

## 7. Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings & completion | `sk-...` |
| `SERPAPI_API_KEY` or `SERPAPI_KEY` | SerpAPI key for external search enrichment | `abcdef123...` |
| `PINECONE_API_KEY` | Pinecone serverless API key | `pcn-...` |
| `PINECONE_INDEX` / `PINECONE_INDEX_NAME` | Target Pinecone index (default if unset) | `fcc-chatbot-index` |
| `PINECONE_ID_STRATEGY` | `url` (deduplicate) or `content` (always grows) | `content` during testing |

You may place these in a `.env` file at the root or use Streamlit secrets (`.streamlit/secrets.toml`). The app loads `.env` first, then overrides with Streamlit secrets where present.

### 7.1 ID Strategy Explained
- `url` (default): Vector IDs = URL + chunk index. Upserts update existing vectors → stable count (preferred for production de-duplication).
- `content`: Vector IDs = hash(content). Every unique chunk grows the index even if from same URL → useful for debugging growth and ensuring new embeddings appear.

## 8. Running the Chat Interfaces

### 8.1 CLI (Rapid Testing)
```powershell
python VectordB/ChromaChat2.py
```
Enter a query. If insufficient internal context is found, the system performs external SerpAPI searches, scrapes selected results, embeds them, and reports growth:
```
✅ Added 9 new embeddings. Pinecone total: 1,854
```

### 8.2 Single-Query Harness
```powershell
python VectordB/run_chat_once.py "Wireless Emergency Alerts site:ready.gov -filetype:pdf"
```

### 8.3 Streamlit Web UI
```powershell
streamlit run VectordB/streamlit_app.py
```
Sidebar shows Pinecone stats and new embeddings added per query cycle.

## 9. Crafting Queries that Add New Embeddings

Use site scoping and PDF exclusion to bias toward HTML articles of sufficient length:
```
Wireless Emergency Alerts site:ready.gov -filetype:pdf
IPAWS architecture site:fema.gov -filetype:pdf
emergency alert system security guidance site:cisa.gov -filetype:pdf
Common Alerting Protocol overview site:nist.gov -filetype:pdf
```
Pages below the minimum length threshold (≈200 chars after cleaning) are skipped.

## 10. Migration & Utility Scripts (VectordB)

| Script | Purpose |
|--------|---------|
| `migrate_chroma_to_pinecone.py` | Full migration of legacy local vectors into Pinecone |
| `simple_migrate.py` | Quick start minimal migration example |
| `fix_pinecone_dimension.py` | Resolves dimension mismatch issues (e.g., wrong embedding model used earlier) |
| `embed_and_upload_texts.py` | Batch embeds raw text files into Pinecone |
| `migrate_root_chroma.py` | Moves root Chroma store artifacts into Pinecone |
| `check_root_chroma.py` | Inspect existing Chroma collections for diagnostics |
| `pinecone_chat.py` | Prototype; superseded by `ChromaChat2.py` |
| `check_pinecone_stats.py` | Print index vector counts & dimension |
| `run_chat_once.py` | Non-interactive single query test harness |

## 11. Training Data Improvement

`fixing-jsonl-files.py` and scripts under `archive/preprocessing/` help refine dataset quality:
- Merge partial datasets → consolidated corpus
- Remove malformed prompt/completion pairs
- Ensure consistent JSONL schema for fine-tuning & evaluation

## 12. Fine-Tuning Workflow (Optional)

1. Generate / clean dataset JSONL (prompt/completion pairs)
2. Validate format & size constraints
3. Upload to provider (e.g., OpenAI) via corresponding scripts in `archive/finetuning/`
4. Monitor status with `list_jobs.py`, `check_status.py`
5. Evaluate results using `chat.py` or `individual_finetune_chat.py`

## 13. Troubleshooting

| Issue | Symptom | Resolution |
|-------|---------|------------|
| No new embeddings added | Log shows 0 added despite upserts | Switch `PINECONE_ID_STRATEGY=content` & ensure query surfaces new long pages |
| Dimension mismatch | Pinecone rejects vectors | Confirm model = `text-embedding-3-small`; run `fix_pinecone_dimension.py` if needed |
| Empty responses | Chat returns minimal text | Check OPENAI_API_KEY validity & rate limits; verify embeddings exist (`check_pinecone_stats.py`) |
| SerpAPI failures | External search step skipped | Ensure `SERPAPI_API_KEY` set; fallback logic may reduce enrichment |
| Large binary repo size | Git push rejects >100MB file | Confirm `.gitignore` excludes `chroma_fcc_storage/` artifacts |
| Repeated overwrites | Pinecone count stable | Expected when using `url` strategy; switch to `content` for growth testing |

## 14. Security & Operational Notes
- Keep API keys in `.env` (not committed) or Streamlit secrets.
- SerpAPI and OpenAI incur cost—batch operations strategically.
- Respect robots.txt & fair use when scraping; implement rate limiting if scaling.
- Consider adding caching for repeated external queries.

## 15. Roadmap / Potential Enhancements
- Add evaluation harness comparing RAG vs fine-tuned model responses
- Automated dataset deduplication with semantic similarity thresholds
- Switch to reranker model for improved context ordering
- Integrate structured citation output (source URLs with confidence)
- Add unit tests around chunking & ID generation logic

## 16. Contributing
Fork and open a Pull Request (PR). Include a summary of vector index impact (added / updated embeddings) in description when modifying retrieval logic.

## 17. License
License information not yet specified. Add appropriate license file before wider distribution.

## 18. Quick Start (TL;DR)
```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Environment (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:SERPAPI_API_KEY="..."
$env:PINECONE_API_KEY="pcn-..."
$env:PINECONE_INDEX="fcc-chatbot-index"
$env:PINECONE_ID_STRATEGY="content"  # optional for growth

# Run Streamlit
streamlit run VectordB/streamlit_app.py

# Or CLI
python VectordB/ChromaChat2.py
```

---
**Need help?** Use `run_chat_once.py` with a targeted query to confirm embeddings growth or `check_pinecone_stats.py` to inspect index health.

