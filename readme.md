# Hazm NLP Service Documentation

## Overview

**Hazm NLP Service** provides Persian (Farsi) NLP capabilities via REST API for RAG systems.

---

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   RAG Service   │────────▶│  Hazm Service    │
│   (Port 8000)   │  HTTP   │   (Port 8001)    │
└─────────────────┘         └──────────────────┘
         │                           │
    Qdrant DB              Persian NLP Processing
    BM25 Index            (Normalize, Tokenize, etc.)
```

---

## Quick Start

```bash
cd hazm_service
docker-compose up -d
curl http://localhost:8001/health
```

---

## Configuration

**Docker Compose:**
```python
HAZM_SERVICE_URL = "http://hazm-service:8001"  # Same network
HAZM_SERVICE_URL = "http://localhost:8001"     # RAG local
```

---

## API Endpoints

### 1. Health Check
**GET** `/health`  
**Input:** None  
**Output:** Service status and available components (normalizer, lemmatizer, stemmer, pos_tagger, chunker)

---

### 2. Text Normalization
**POST** `/normalize`  
**Input:** `{"text": "متن فارسي", "persian_style": true, "punctuation_spacing": true, "affix_spacing": true}`  
**Output:** `{"original": "...", "normalized": "..."}`  
**Purpose:** Converts Arabic chars to Persian (ي→ی, ك→ک), fixes spacing

---

### 3. Batch Normalization
**POST** `/normalize/batch`  
**Input:** `{"texts": ["متن اول", "متن دوم"]}`  
**Output:** `{"results": [{"original": "...", "normalized": "..."}], "count": 2}`  
**Purpose:** Normalize multiple texts efficiently

---

### 4. Word Tokenization
**POST** `/tokenize/words`  
**Input:** `{"text": "کتاب‌های من", "join_verb_parts": true, "join_noun_parts": true}`  
**Output:** `{"tokens": ["کتاب‌های", "من"], "token_count": 2}`  
**Purpose:** Split text into words, handle Persian compounds

---

### 5. Sentence Tokenization
**POST** `/tokenize/sentences`  
**Input:** `{"text": "جمله اول. جمله دوم؟"}`  
**Output:** `{"sentences": [{"index": 0, "text": "جمله اول.", "word_count": 2, "char_count": 10}], "sentence_count": 2}`  
**Purpose:** Split into sentences with metadata (word/char counts)

---

### 6. Lemmatization
**POST** `/lemmatize`  
**Input:** `{"words": ["کتاب‌ها", "می‌روم", "بزرگ‌ترین"]}`  
**Output:** `{"results": [{"word": "کتاب‌ها", "lemma": "کتاب"}], "count": 3}`  
**Purpose:** Convert words to base form (کتاب‌ها → کتاب)

---

### 7. Stemming
**POST** `/stem`  
**Input:** `{"words": ["کتابخانه", "کتاب‌فروشی"]}`  
**Output:** `{"results": [{"word": "کتابخانه", "stem": "کتاب"}], "count": 2}`  
**Purpose:** Extract root morpheme (more aggressive than lemmatization)

---

### 8. Entity Extraction
**POST** `/extract/entities`  
**Input:** `{"text": "تماس: 09123456789 ایمیل: test@example.com"}`  
**Output:** `{"entities": {"numbers": ["09123456789"], "urls": [], "emails": ["test@example.com"], "dates": []}, "total_entities": 2}`  
**Purpose:** Extract phones, emails, URLs, dates, Persian months

---

### 9. Title Extraction ⭐
**POST** `/extract/title`  
**Input:** `{"text": "کتاب‌های دانشگاهی برای دانشجویان کامپیوتر"}`  
**Output:** `{"generated_title": "کتاب - دانشگاه - کامپیوتر", "top_lemmas": ["کتاب", "دانشگاه", "کامپیوتر"], "lemma_frequencies": {...}}`  
**Purpose:** Generate chunk title from top 3 frequent lemmas (excludes stopwords)

---

### 10. Advanced Sentence Splitting
**POST** `/split/sentences/advanced`  
**Input:** `{"text": "جمله اول. جمله طولانی‌تر.", "preserve_punctuation": true}`  
**Output:** `{"sentences": [{"index": 0, "text": "...", "word_count": 2, "char_count": 10, "starts_with_capital": false}], "total_sentences": 2}`  
**Purpose:** Detailed sentence analysis with metadata

---

### 11. Semantic Chunking ⭐ (Most Important)
**POST** `/chunk/semantic`  
**Input:** `{"text": "فصل اول...\n\nفصل دوم...", "min_chunk_size": 100, "max_chunk_size": 150}`  
**Output:** `{"chunks": [{"text": "...", "word_count": 125, "index": 0, "prev_context": "", "next_context": "..."}], "total_chunks": 2, "statistics": {...}}`  
**Purpose:** Split documents into RAG-ready chunks (respects sentence boundaries, adds context windows)

---

### 12. POS Tagging ⚠️
**POST** `/pos_tag`  
**Input:** `{"text": "کتاب‌های جدید را گذاشتند"}`  
**Output:** `{"tags": [{"word": "کتاب‌های", "pos": "N"}, {"word": "جدید", "pos": "ADJ"}], "token_count": 5}`  
**Purpose:** Identify word types (N=Noun, V=Verb, ADJ=Adjective, etc.) - Requires Wapiti

---

### 13. Syntactic Chunking ⚠️
**POST** `/chunk/syntactic`  
**Input:** `{"text": "دانشجویان دانشگاه تهران در کتابخانه مطالعه می‌کنند"}`  
**Output:** `{"sentences": [{"sentence": "...", "tokens": [...], "tagged": [...], "tree": "..."}], "sentence_count": 1}`  
**Purpose:** Extract noun/verb phrases (NP, VP, PP) - Requires Wapiti + Chunker

---

### 14. Complete Text Analysis
**POST** `/analyze/text`  
**Input:** `{"text": "کتاب‌های جدید در کتابخانه"}`  
**Output:** `{"statistics": {"char_count": 39, "word_count": 6, "unique_word_count": 6, "avg_word_length": 5.5}, "sentences": [...], "words": [...], "lemmas": [...], "stems": [...]}`  
**Purpose:** All-in-one analysis (stats + tokenization + lemmatization + stemming)

---

### 15. Document Preprocessing
**POST** `/preprocess/document`  
**Input:** `{"text": "جمله اول. جمله دوم."}`  
**Output:** `{"processed_sentences": [{"original": "...", "words": [...], "lemmas": [...]}], "all_words": [...], "all_lemmas": [...], "statistics": {...}}`  
**Purpose:** Full preprocessing pipeline (normalize + tokenize + lemmatize per sentence)

---

## Integration Example

```python
import httpx

HAZM_SERVICE_URL = "http://localhost:8001"

async def call_hazm_service(endpoint: str, data: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{HAZM_SERVICE_URL}{endpoint}", json=data)
        return response.json()

# Normalize text
result = await call_hazm_service("/normalize", {"text": "متن فارسي"})

# Semantic chunking
chunks = await call_hazm_service("/chunk/semantic", {
    "text": document_text,
    "min_chunk_size": 100,
    "max_chunk_size": 150
})

# Extract title for chunk
title = await call_hazm_service("/extract/title", {"text": chunk_text})
```

---

## Common Workflows

**Document Upload → Embedding:**
```
1. Extract text from PDF/DOCX
2. POST /normalize → clean text
3. POST /chunk/semantic → create chunks
4. For each chunk:
   - POST /extract/title → generate title
   - POST /extract/entities → find metadata
5. Generate embeddings
6. Store in Qdrant with metadata
```

**Search Query:**
```
1. POST /normalize → clean query
2. Generate embedding
3. Search Qdrant + BM25
4. Return results with chunk titles
```

---

## Performance

| Endpoint | Text Size | Response Time |
|----------|-----------|---------------|
| `/normalize` | 1KB | ~10ms |
| `/tokenize/words` | 1KB | ~15ms |
| `/extract/title` | 1KB | ~30ms |
| `/chunk/semantic` | 10KB | ~100ms |
| `/chunk/semantic` | 100KB | ~500ms |

---

## Error Handling

**POS/Chunker unavailable:**
```json
{"detail": "POS Tagger not available"}
```
→ Check `/health` - these require Wapiti models

**Timeout:**
```json
{"detail": "Request timeout"}
```
→ Text too large, split before processing

**Connection error:**
→ Verify `HAZM_SERVICE_URL` and service status

---

## Testing

```bash
# Health check
curl http://localhost:8001/health | jq

# Normalize
curl -X POST http://localhost:8001/normalize \
  -H "Content-Type: application/json" \
  -d '{"text":"متن فارسي"}'

# Extract title
curl -X POST http://localhost:8001/extract/title \
  -H "Content-Type: application/json" \
  -d '{"text":"کتاب‌های دانشگاهی کامپیوتر"}'

# Semantic chunking
curl -X POST http://localhost:8001/chunk/semantic \
  -H "Content-Type: application/json" \
  -d '{"text":"پاراگراف اول.\n\nپاراگراف دوم.","min_chunk_size":50,"max_chunk_size":150}'
```

---

## Summary Table

| Feature | Endpoint | Primary Use | Requires |
|---------|----------|-------------|----------|
| Health Check | `GET /health` | Monitoring | - |
| Normalize | `POST /normalize` | Clean Persian text | - |
| Tokenize Words | `POST /tokenize/words` | BM25 indexing | - |
| Tokenize Sentences | `POST /tokenize/sentences` | Sentence splitting | - |
| **Semantic Chunk** | `POST /chunk/semantic` | **Document chunking** | - |
| Lemmatize | `POST /lemmatize` | Search expansion | - |
| Stem | `POST /stem` | Root extraction | - |
| **Extract Title** | `POST /extract/title` | **Chunk titles** | - |
| Extract Entities | `POST /extract/entities` | Metadata enrichment | - |
| Advanced Split | `POST /split/sentences/advanced` | Detailed analysis | - |
| Analyze Text | `POST /analyze/text` | Full analysis | - |
| Preprocess Doc | `POST /preprocess/document` | Complete pipeline | - |
| POS Tag | `POST /pos_tag` | Grammar analysis | Wapiti ⚠️ |
| Syntactic Chunk | `POST /chunk/syntactic` | Phrase extraction | Wapiti ⚠️ |

**Most Important for RAG:**
1. ⭐ `/chunk/semantic` - Primary document chunking
2. ⭐ `/normalize` - Text preprocessing
3. ⭐ `/extract/title` - Chunk titles for UX
4. `/extract/entities` - Metadata enrichment
5. `/tokenize/words` - BM25 indexing

---

## Troubleshooting

**Service won't start:**
```bash
docker logs hazm_nlp_service
```

**Connection refused:**
- Check service: `docker ps | grep hazm`
- Verify URL: Use service name in Docker network
- Test: `curl http://localhost:8001/health`

**Slow performance:**
- Large text (>100KB)? Split before processing
- Increase Docker memory
- Use caching for repeated calls

---

## FAQ

**Q: POS/Chunker not available?**  
A: These require Wapiti models. Check `/health` - core features work without them.

**Q: Best chunk size?**  
A: `min_chunk_size=100, max_chunk_size=150` for most RAG use cases.

**Q: How to customize stopwords?**  
A: Edit `stop_words` set in `extract_title_from_text` function in main.py.

**Q: Run without Docker?**  
A: `pip install hazm fastapi uvicorn && python main.py` (Python 3.11 required)

---

## Payload Example (Stored in Qdrant)

```json
{
  "text": "کتاب‌های علمی در دانشگاه",
  "text_normalized": "کتاب‌های علمی در دانشگاه",
  "chunk_index": 0,
  "prev_context": "",
  "next_context": "...",
  "word_count": 125,
  "chunk_title": "کتاب - علم - دانشگاه",
  "top_lemmas": ["کتاب", "علم", "دانشگاه"],
  "entities": {
    "numbers": [],
    "emails": [],
    "urls": [],
    "dates": []
  },
  "file_id": "abc-123",
  "source": "document.pdf"
}
```

---

## License

MIT License - Free to use and modify

---

## Support

- Check logs: `docker logs hazm_nlp_service`
- Verify health: `curl http://localhost:8001/health`
- Documentation: This README

---

**End of Documentation**