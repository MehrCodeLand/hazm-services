# Hazm NLP Service Documentation

## Overview

**Hazm NLP Service** is a microservice providing Persian (Farsi) natural language processing capabilities via REST API. It's designed to work alongside RAG systems to provide language-specific text processing that generic NLP libraries cannot handle effectively.

---

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   RAG Service   │────────▶│  Hazm Service    │
│   (Port 8000)   │  HTTP   │   (Port 8001)    │
└─────────────────┘         └──────────────────┘
         │                           │
         │                           │
    Qdrant DB              Persian NLP Processing
    BM25 Index            (Normalize, Tokenize, etc.)
```

---

## Why This Service Exists

### Problem
- NLTK, spaCy, and other NLP libraries are optimized for English
- Persian has unique challenges:
  - Different character forms (ی/ي, ک/ك)
  - Arabic vs Persian characters mixed in text
  - Different punctuation (؟ ؛)
  - Right-to-left text direction
  - Complex morphology

### Solution
- Dedicated Persian NLP service using **Hazm library**
- Microservice architecture allows Python 3.11 (Hazm requirement) separate from main RAG service
- Centralized Persian text processing for consistent results

---

## Installation & Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.11 (for local development)

### Quick Start

```bash
cd hazm_service
docker-compose up -d
```

Check service health:
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "hazm-nlp-service",
  "components": {
    "normalizer": true,
    "lemmatizer": true,
    "stemmer": true,
    "pos_tagger": false,
    "chunker": false
  }
}
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Check service status and available components

**Input:** None

**Output:**
```json
{
  "status": "healthy",
  "service": "hazm-nlp-service",
  "components": {
    "normalizer": true,
    "lemmatizer": true,
    "stemmer": true,
    "pos_tagger": false,
    "chunker": false
  }
}
```

**Use Case:** Container health monitoring, startup verification

---

### 2. Text Normalization

**Endpoint:** `POST /normalize`

**Purpose:** Standardize Persian text by fixing character variations, spacing, and encoding issues

**Input:**
```json
{
  "text": "این یک متن فارسي است با حروف عربي مثل ك و ي",
  "persian_style": true,
  "punctuation_spacing": true,
  "affix_spacing": true
}
```

**Output:**
```json
{
  "original": "این یک متن فارسي است با حروف عربي مثل ك و ي",
  "normalized": "این یک متن فارسی است با حروف عربی مثل ک و ی",
  "char_changes": 4
}
```

**What It Does:**
- Converts `ي` (Arabic Yeh) → `ی` (Persian Yeh)
- Converts `ك` (Arabic Kaf) → `ک` (Persian Kaf)
- Fixes spacing around punctuation
- Removes zero-width characters
- Standardizes quotation marks

**Use Case:** Pre-processing before embedding generation, search query normalization

---

### 3. Batch Normalization

**Endpoint:** `POST /normalize/batch`

**Purpose:** Normalize multiple texts efficiently

**Input:**
```json
{
  "texts": [
    "متن اول با حروف عربي",
    "متن دوم با فاصله‌گذاري نادرست",
    "متن سوم"
  ]
}
```

**Output:**
```json
{
  "results": [
    {
      "original": "متن اول با حروف عربي",
      "normalized": "متن اول با حروف عربی"
    },
    {
      "original": "متن دوم با فاصله‌گذاري نادرست",
      "normalized": "متن دوم با فاصله‌گذاری نادرست"
    },
    {
      "original": "متن سوم",
      "normalized": "متن سوم"
    }
  ],
  "count": 3
}
```

**Use Case:** Processing multiple document chunks at once

---

### 4. Word Tokenization

**Endpoint:** `POST /tokenize/words`

**Purpose:** Split text into individual words (tokens)

**Input:**
```json
{
  "text": "کتاب‌های من روی میز هستند.",
  "join_verb_parts": true,
  "join_noun_parts": true
}
```

**Output:**
```json
{
  "text": "کتاب‌های من روی میز هستند.",
  "normalized": "کتاب‌های من روی میز هستند.",
  "tokens": ["کتاب‌های", "من", "روی", "میز", "هستند", "."],
  "token_count": 6
}
```

**What It Does:**
- Handles Persian compound words (کتاب‌ها)
- Preserves zero-width non-joiner (ZWNJ) connections
- Separates punctuation correctly
- Recognizes verb and noun affixes

**Use Case:** BM25 indexing, token counting for chunks, keyword extraction

---

### 5. Sentence Tokenization

**Endpoint:** `POST /tokenize/sentences`

**Purpose:** Split text into sentences - **CRITICAL FOR CHUNKING**

**Input:**
```json
{
  "text": "این جمله اول است. این جمله دوم است؟ و این جمله سوم!"
}
```

**Output:**
```json
{
  "text": "این جمله اول است. این جمله دوم است؟ و این جمله سوم!",
  "normalized": "این جمله اول است. این جمله دوم است؟ و این جمله سوم!",
  "sentences": [
    {
      "index": 0,
      "text": "این جمله اول است.",
      "word_count": 4,
      "char_count": 17
    },
    {
      "index": 1,
      "text": "این جمله دوم است؟",
      "word_count": 4,
      "char_count": 17
    },
    {
      "index": 2,
      "text": "و این جمله سوم!",
      "word_count": 4,
      "char_count": 15
    }
  ],
  "sentence_count": 3
}
```

**What It Does:**
- Detects Persian sentence boundaries (. ؟ ! ؛)
- Handles abbreviations correctly
- Recognizes Persian punctuation patterns
- Maintains sentence integrity

**Use Case:** **This is what replaces NLTK in your RAG chunking** - ensures chunks don't break mid-sentence

---

### 6. Lemmatization

**Endpoint:** `POST /lemmatize`

**Purpose:** Convert words to their base/dictionary form

**Input:**
```json
{
  "words": ["کتاب‌ها", "می‌روم", "رفتند", "بزرگ‌ترین"]
}
```

**Output:**
```json
{
  "results": [
    {
      "word": "کتاب‌ها",
      "normalized": "کتاب‌ها",
      "lemma": "کتاب"
    },
    {
      "word": "می‌روم",
      "normalized": "می‌روم",
      "lemma": "رفت"
    },
    {
      "word": "رفتند",
      "normalized": "رفتند",
      "lemma": "رفت"
    },
    {
      "word": "بزرگ‌ترین",
      "normalized": "بزرگ‌ترین",
      "lemma": "بزرگ"
    }
  ],
  "count": 4
}
```

**What It Does:**
- Removes plural markers (ها، ان)
- Converts verbs to infinitive form
- Removes comparative/superlative markers (تر، ترین)
- Handles verb conjugations

**Use Case:** Improving search recall (searching "کتاب" finds "کتاب‌ها"), text analysis

---

### 7. Stemming

**Endpoint:** `POST /stem`

**Purpose:** Reduce words to root form (more aggressive than lemmatization)

**Input:**
```json
{
  "words": ["کتاب‌های", "کتابخانه", "کتاب‌فروشی"]
}
```

**Output:**
```json
{
  "results": [
    {
      "word": "کتاب‌های",
      "normalized": "کتاب‌های",
      "stem": "کتاب"
    },
    {
      "word": "کتابخانه",
      "normalized": "کتابخانه",
      "stem": "کتاب"
    },
    {
      "word": "کتاب‌فروشی",
      "normalized": "کتاب‌فروشی",
      "stem": "کتاب"
    }
  ],
  "count": 3
}
```

**What It Does:**
- Extracts root morpheme
- More aggressive than lemmatization
- May produce non-words

**Use Case:** Search expansion, finding related terms

---

### 8. Entity Extraction

**Endpoint:** `POST /extract/entities`

**Purpose:** Extract structured information (numbers, dates, emails, URLs)

**Input:**
```json
{
  "text": "تماس: 09123456789 ایمیل: test@example.com تاریخ: 15 فروردین 1403 قیمت: 1,500,000 تومان سایت: https://example.com"
}
```

**Output:**
```json
{
  "text": "تماس: 09123456789 ایمیل: test@example.com...",
  "entities": {
    "numbers": ["09123456789", "1,500,000"],
    "urls": ["https://example.com"],
    "emails": ["test@example.com"],
    "dates": ["فروردین"]
  },
  "total_entities": 5
}
```

**What It Does:**
- Extracts phone numbers, prices, IDs
- Finds URLs and email addresses
- Detects Persian month names
- Pattern-based extraction

**Use Case:** Metadata enrichment for chunks, filtering by entity types

---

### 9. Advanced Sentence Splitting

**Endpoint:** `POST /split/sentences/advanced`

**Purpose:** Detailed sentence analysis with metadata

**Input:**
```json
{
  "text": "جمله اول. جمله دوم با جزئیات بیشتر و طولانی‌تر است. جمله کوتاه.",
  "preserve_punctuation": true
}
```

**Output:**
```json
{
  "original": "جمله اول. جمله دوم...",
  "normalized": "جمله اول. جمله دوم...",
  "sentences": [
    {
      "index": 0,
      "text": "جمله اول.",
      "word_count": 2,
      "char_count": 10,
      "starts_with_capital": false
    },
    {
      "index": 1,
      "text": "جمله دوم با جزئیات بیشتر و طولانی‌تر است.",
      "word_count": 7,
      "char_count": 42,
      "starts_with_capital": false
    },
    {
      "index": 2,
      "text": "جمله کوتاه.",
      "word_count": 2,
      "char_count": 11,
      "starts_with_capital": false
    }
  ],
  "total_sentences": 3
}
```

**Use Case:** Analyzing text structure, identifying long/short sentences for adaptive chunking

---

### 10. Semantic Chunking ⭐ (Most Important for RAG)

**Endpoint:** `POST /chunk/semantic`

**Purpose:** **Split documents into semantic chunks for RAG embedding** - This is the core replacement for your NLTK chunking

**Input:**
```json
{
  "text": "فصل اول: مقدمه\n\nشرکت ما در سال 1990 تاسیس شد. هدف اصلی ارائه خدمات با کیفیت است.\n\nفصل دوم: تاریخچه\n\nدر سال 1995 اولین شعبه افتتاح شد. امروز بیش از 100 شعبه داریم.",
  "min_chunk_size": 50,
  "max_chunk_size": 150
}
```

**Output:**
```json
{
  "original_text": "فصل اول: مقدمه\n\nشرکت ما...",
  "normalized_text": "فصل اول: مقدمه\n\nشرکت ما...",
  "chunks": [
    {
      "text": "فصل اول: مقدمه\n\nشرکت ما در سال 1990 تاسیس شد. هدف اصلی ارائه خدمات با کیفیت است.",
      "word_count": 15,
      "paragraph_count": 2,
      "index": 0,
      "prev_context": "",
      "next_context": "فصل دوم: تاریخچه\n\nدر سال 1995..."
    },
    {
      "text": "فصل دوم: تاریخچه\n\nدر سال 1995 اولین شعبه افتتاح شد. امروز بیش از 100 شعبه داریم.",
      "word_count": 16,
      "paragraph_count": 2,
      "index": 1,
      "prev_context": "...هدف اصلی ارائه خدمات با کیفیت است.",
      "next_context": ""
    }
  ],
  "total_chunks": 2,
  "min_chunk_size": 50,
  "max_chunk_size": 150
}
```

**What It Does:**
1. Normalizes text
2. Splits into paragraphs
3. Splits paragraphs into sentences
4. Combines sentences into chunks respecting size limits
5. Never breaks mid-sentence
6. Adds prev/next context for better embeddings
7. Maintains paragraph boundaries when possible

**Use Case:** **PRIMARY USE** - This is what your RAG service calls instead of the NLTK sentence splitter

**Algorithm:**
```
Text → Normalize → Split Paragraphs
                         ↓
                Split into Sentences (Persian-aware)
                         ↓
            Combine Sentences into Chunks
            (respecting min/max size)
                         ↓
            Add Context (prev/next chunks)
                         ↓
            Return Chunks Ready for Embedding
```

---

### 11. Complete Text Analysis

**Endpoint:** `POST /analyze/text`

**Purpose:** Comprehensive analysis of text - all operations at once

**Input:**
```json
{
  "text": "کتاب‌های جدید در کتابخانه قرار گرفتند."
}
```

**Output:**
```json
{
  "original": "کتاب‌های جدید در کتابخانه قرار گرفتند.",
  "normalized": "کتاب‌های جدید در کتابخانه قرار گرفتند.",
  "statistics": {
    "char_count": 39,
    "word_count": 6,
    "unique_word_count": 6,
    "sentence_count": 1,
    "avg_word_length": 5.5,
    "avg_sentence_length": 6.0
  },
  "sentences": ["کتاب‌های جدید در کتابخانه قرار گرفتند."],
  "words": ["کتاب‌های", "جدید", "در", "کتابخانه", "قرار", "گرفتند"],
  "lemmas": ["کتاب", "جدید", "در", "کتابخانه", "قرار", "گرفت"],
  "stems": ["کتاب", "جدید", "در", "کتاب", "قرار", "گرفت"],
  "pos_tags": []
}
```

**Use Case:** Document statistics, quality assessment, exploratory analysis

---

### 12. Document Preprocessing

**Endpoint:** `POST /preprocess/document`

**Purpose:** Full preprocessing pipeline for document ingestion

**Input:**
```json
{
  "text": "جمله اول. جمله دوم با کلمات بیشتری دارد. جمله سوم کوتاه است."
}
```

**Output:**
```json
{
  "original": "جمله اول. جمله دوم...",
  "normalized": "جمله اول. جمله دوم...",
  "processed_sentences": [
    {
      "original": "جمله اول.",
      "words": ["جمله", "اول", "."],
      "lemmas": ["جمله", "اول", "."]
    },
    {
      "original": "جمله دوم با کلمات بیشتری دارد.",
      "words": ["جمله", "دوم", "با", "کلمات", "بیشتری", "دارد", "."],
      "lemmas": ["جمله", "دوم", "با", "کلمه", "بیشتر", "داشت", "."]
    },
    {
      "original": "جمله سوم کوتاه است.",
      "words": ["جمله", "سوم", "کوتاه", "است", "."],
      "lemmas": ["جمله", "سوم", "کوتاه", "بود", "."]
    }
  ],
  "all_words": ["جمله", "اول", ".", "جمله", "دوم", ...],
  "all_lemmas": ["جمله", "اول", ".", "جمله", "دوم", ...],
  "statistics": {
    "sentence_count": 3,
    "word_count": 15,
    "unique_lemmas": 12
  }
}
```

**Use Case:** Pre-processing for BM25 indexing, feature extraction

---

## Integration with RAG Service

### How RAG Calls Hazm

**In your RAG code (main.py):**

```python
import httpx

HAZM_SERVICE_URL = "http://hazm-service:8001"

async def call_hazm_service(endpoint: str, data: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{HAZM_SERVICE_URL}{endpoint}", json=data)
        return response.json()

async def semantic_chunker(text: str, min_chunk_size: int = 100, max_chunk_size: int = 150):
    hazm_result = await call_hazm_service("/chunk/semantic", {
        "text": text,
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size
    })
    
    if hazm_result and "chunks" in hazm_result:
        return hazm_result["chunks"]
    
```

---

## Common Workflows

### Workflow 1: Document Upload → Chunking → Embedding

```
1. RAG receives PDF/DOCX upload
         ↓
2. RAG extracts text
         ↓
3. RAG calls Hazm: POST /chunk/semantic
         ↓
4. Hazm returns semantic chunks
         ↓
5. RAG generates embeddings for each chunk
         ↓
6. RAG stores in Qdrant + BM25 index
```

### Workflow 2: Search Query Processing

```
1. User sends search query
         ↓
2. RAG calls Hazm: POST /normalize
         ↓
3. Hazm normalizes query text
         ↓
4. RAG generates query embedding
         ↓
5. RAG searches Qdrant (vector) + BM25 (lexical)
         ↓
6. Return results
```

### Workflow 3: BM25 Index Building

```
1. RAG has chunk text
         ↓
2. RAG calls Hazm: POST /tokenize/words
         ↓
3. Hazm returns tokens
         ↓
4. RAG builds BM25 index with tokens
         ↓
5. Index ready for lexical search
```

---

## Performance Considerations

### Response Times (Approximate)

| Endpoint | Text Size | Response Time |
|----------|-----------|---------------|
| `/normalize` | 1KB | ~10ms |
| `/tokenize/words` | 1KB | ~15ms |
| `/tokenize/sentences` | 1KB | ~20ms |
| `/chunk/semantic` | 10KB | ~100ms |
| `/chunk/semantic` | 100KB | ~500ms |

### Optimization Tips

1. **Use batch endpoints** when processing multiple texts
2. **Cache normalization** results for identical texts
3. **Call `/chunk/semantic` directly** instead of separate tokenization calls
4. **Monitor service health** - restart if components fail

---

## Error Handling

### Common Errors

**Service Unavailable (503):**
```json
{
  "detail": "POS Tagger not available"
}
```
**Solution:** This is expected if libwapiti didn't build. Core features still work.

**Timeout Error:**
```json
{
  "detail": "Request timeout"
}
```
**Solution:** Text too large. Split into smaller chunks before calling.

**Invalid Text:**
```json
{
  "detail": "Normalization error: empty text"
}
```
**Solution:** Validate text is non-empty before calling.

---

## Testing

### Quick Test Script

```bash
#!/bin/bash

echo "Testing Hazm Service..."

echo "\n1. Health Check:"
curl -s http://localhost:8001/health | jq

echo "\n2. Normalize Text:"
curl -s -X POST http://localhost:8001/normalize \
  -H "Content-Type: application/json" \
  -d '{"text":"این یک متن فارسي است"}' | jq

echo "\n3. Tokenize Sentences:"
curl -s -X POST http://localhost:8001/tokenize/sentences \
  -H "Content-Type: application/json" \
  -d '{"text":"جمله اول. جمله دوم."}' | jq

echo "\n4. Semantic Chunking:"
curl -s -X POST http://localhost:8001/chunk/semantic \
  -H "Content-Type: application/json" \
  -d '{"text":"پاراگراف اول با چند جمله.\n\nپاراگراف دوم.","min_chunk_size":50,"max_chunk_size":150}' | jq
```

---

## Monitoring

### Docker Health Check

Service includes built-in health check:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### View Logs

```bash
docker logs -f hazm_nlp_service
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
docker logs hazm_nlp_service
```

**Common issues:**
- Port 8001 already in use → Change port in docker-compose.yml
- Out of memory → Increase Docker memory limit
- Build failed → Check Dockerfile dependencies

### Service Running But Not Responding

**Test connectivity:**
```bash
docker exec -it hazm_nlp_service curl http://localhost:8001/health
```

**Restart service:**
```bash
docker-compose restart hazm-service
```

---

## Summary Table

| Feature | Endpoint | Input | Output | RAG Use Case |
|---------|----------|-------|--------|--------------|
| Health Check | `GET /health` | None | Status | Monitoring |
| Normalize | `POST /normalize` | Persian text | Clean text | Query/doc preprocessing |
| Tokenize Words | `POST /tokenize/words` | Text | Word list | BM25 indexing |
| Tokenize Sentences | `POST /tokenize/sentences` | Text | Sentence list | Chunk boundary detection |
| **Semantic Chunk** | `POST /chunk/semantic` | **Document** | **Chunks** | **Primary chunking** |
| Lemmatize | `POST /lemmatize` | Words | Base forms | Search expansion |
| Extract Entities | `POST /extract/entities` | Text | Entities | Metadata enrichment |

**Most Important Endpoints for RAG:**
1. ⭐ `/chunk/semantic` - Replaces NLTK chunking
2. ⭐ `/normalize` - Cleans all Persian text
3. ⭐ `/tokenize/words` - For BM25 indexing

---

## License

MIT License - Free to use and modify

## Support

For issues or questions:
- Check logs: `docker logs hazm_nlp_service`
- Verify health: `curl http://localhost:8001/health`
- Review this README for usage examples