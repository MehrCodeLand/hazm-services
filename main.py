from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
from hazm import Normalizer, word_tokenize, sent_tokenize, Lemmatizer, Stemmer, POSTagger
from hazm import Chunker, DependencyParser
import re




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Hazm NLP Service", version="1.0.0")

normalizer = Normalizer()
lemmatizer = Lemmatizer()
stemmer = Stemmer()
tagger = POSTagger(model='pos_tagger.model')      
chunker = Chunker(model='chunker.model') 


try:
    tagger = POSTagger(model='pos_tagger.model')
    logger.info("POS Tagger loaded successfully")
except:
    tagger = None
    logger.warning("POS Tagger model not found, POS tagging will be disabled")

try:
    chunker = Chunker(model='chunker.model')
    logger.info("Chunker loaded successfully")
except:
    chunker = None
    logger.warning("Chunker model not found, chunking will be disabled")


class TextRequest(BaseModel):
    text: str


class TextListRequest(BaseModel):
    texts: List[str]


class NormalizationRequest(BaseModel):
    text: str
    persian_style: bool = True
    punctuation_spacing: bool = True
    affix_spacing: bool = True


class TokenizationRequest(BaseModel):
    text: str
    join_verb_parts: bool = True
    join_noun_parts: bool = True


class SentenceRequest(BaseModel):
    text: str


class LemmatizeRequest(BaseModel):
    words: List[str]


class StemRequest(BaseModel):
    words: List[str]


class POSRequest(BaseModel):
    text: str


class ChunkRequest(BaseModel):
    text: str


class EntityRequest(BaseModel):
    text: str


class SemanticChunkRequest(BaseModel):
    text: str
    min_chunk_size: int = 100
    max_chunk_size: int = 150


class SentenceSplitRequest(BaseModel):
    text: str
    preserve_punctuation: bool = True


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "hazm-nlp-service",
        "components": {
            "normalizer": True,
            "lemmatizer": True,
            "stemmer": True,
            "pos_tagger": tagger is not None,
            "chunker": chunker is not None
        }
    }


@app.post("/normalize")
async def normalize_text(request: NormalizationRequest):
    try:
        normalized = normalizer.normalize(request.text)
        
        if not request.persian_style:
            normalized = normalized.replace('ی', 'ي').replace('ک', 'ك')
        
        return {
            "original": request.text,
            "normalized": normalized,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization error: {str(e)}")


@app.post("/normalize/batch")
async def normalize_batch(request: TextListRequest):
    try:
        results = []
        for text in request.texts:
            normalized = normalizer.normalize(text)
            results.append({
                "original": text,
                "normalized": normalized
            })
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch normalization error: {str(e)}")


@app.post("/tokenize/words")
async def tokenize_words(request: TokenizationRequest):
    try:
        normalized = normalizer.normalize(request.text)
        tokens = word_tokenize(normalized)
        
        return {
            "text": request.text,
            "normalized": normalized,
            "tokens": tokens,
            "token_count": len(tokens)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization error: {str(e)}")


@app.post("/tokenize/sentences")
async def tokenize_sentences(request: SentenceRequest):
    try:
        normalized = normalizer.normalize(request.text)
        sentences = sent_tokenize(normalized)
        
        sentence_details = []
        for idx, sent in enumerate(sentences):
            words = word_tokenize(sent)
            sentence_details.append({
                "index": idx,
                "text": sent,
                "word_count": len(words),
                "char_count": len(sent)
            })
        
        return {
            "text": request.text,
            "normalized": normalized,
            "sentences": sentence_details,
            "sentence_count": len(sentences)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentence tokenization error: {str(e)}")


@app.post("/lemmatize")
async def lemmatize_words(request: LemmatizeRequest):
    try:
        results = []
        for word in request.words:
            normalized = normalizer.normalize(word)
            lemma = lemmatizer.lemmatize(normalized)
            results.append({
                "word": word,
                "normalized": normalized,
                "lemma": lemma
            })
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lemmatization error: {str(e)}")


@app.post("/stem")
async def stem_words(request: StemRequest):
    try:
        results = []
        for word in request.words:
            normalized = normalizer.normalize(word)
            stem = stemmer.stem(normalized)
            results.append({
                "word": word,
                "normalized": normalized,
                "stem": stem
            })
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stemming error: {str(e)}")


@app.post("/pos_tag")
async def pos_tag(request: POSRequest):
    try:
        if tagger is None:
            raise HTTPException(status_code=503, detail="POS Tagger not available")
        
        normalized = normalizer.normalize(request.text)
        tokens = word_tokenize(normalized)
        tags = tagger.tag(tokens)
        
        tagged_results = []
        for word, tag in tags:
            tagged_results.append({
                "word": word,
                "pos": tag
            })
        
        return {
            "text": request.text,
            "normalized": normalized,
            "tags": tagged_results,
            "token_count": len(tagged_results)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"POS tagging error: {str(e)}")


@app.post("/chunk/syntactic")
async def syntactic_chunk(request: ChunkRequest):
    try:
        if chunker is None:
            raise HTTPException(status_code=503, detail="Chunker not available")
        
        if tagger is None:
            raise HTTPException(status_code=503, detail="POS Tagger not available")

        # Step 1: Normalize text
        normalized = normalizer.normalize(request.text)

        # Step 2: Split into sentences
        sentences = sent_tokenize(normalized)
        
        # Step 3: Process each sentence
        all_tagged = []
        all_trees = []
        
        for sent in sentences:
            # Tokenize sentence
            tokens = word_tokenize(sent)
            
            # POS tag the tokens
            tagged = tagger.tag(tokens)  # [('من', 'PRON'), ('به', 'ADP'), ...]
            all_tagged.append(tagged)
            
            # Parse the tagged sentence - chunker.parse expects list of tagged sentences
            tree = chunker.parse(tagged)  # Pass tagged directly, not [tagged]
            all_trees.append(str(tree))
        
        # Step 4: Return structured result
        return {
            "text": request.text,
            "normalized": normalized,
            "sentences": [
                {
                    "sentence": sent,
                    "tokens": [word for word, pos in tagged],
                    "tagged": tagged,
                    "tree": tree
                }
                for sent, tagged, tree in zip(sentences, all_tagged, all_trees)
            ],
            "sentence_count": len(sentences)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking error: {str(e)}")


@app.post("/extract/entities")
async def extract_entities(request: EntityRequest):
    try:
        normalized = normalizer.normalize(request.text)
        text_list = request.text.split()
        # tokens = word_tokenize(request.text)

        entities = {
            "numbers": [],
            "urls": [],
            "emails": [],
            "dates": []
        }
        
        number_pattern = r'\d+(?:[.,]\d+)*'
        url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        

        for token in text_list:
            if re.match(number_pattern, token):
                entities["numbers"].append(token)
            elif re.match(url_pattern, token):
                entities["urls"].append(token)
            elif re.match(email_pattern, token):
                entities["emails"].append(token)
        
        persian_months = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 
                         'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند']
        
        for month in persian_months:
            if month in normalized:
                entities["dates"].append(month)
        
        return {
            "text": request.text,
            "entities": entities,
            "total_entities": sum(len(v) for v in entities.values()),
            # "words": tokens,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity extraction error: {str(e)}")


@app.post("/extract/title")
async def extract_title_from_text(request: TextRequest):
    try:
        normalized = normalizer.normalize(request.text)
        words = word_tokenize(normalized)
        
        lemmas = [lemmatizer.lemmatize(w) for w in words if len(w) > 2]
        
        from collections import Counter
        lemma_freq = Counter(lemmas)
        
        stop_words = {
            'است', 'شد', 'شده', 'می', 'را', 'به', 'از', 'در', 'که', 'این', 
            'آن', 'با', 'برای', 'یک', 'هم', 'خود', 'تا', 'کرد', 'بر', 'هر',
            'نیز', 'اما', 'یا', 'چه', 'و', 'ای', 'دارد', 'داشت', 'کند',
            'شود', 'گفت', 'کرده', 'دو', 'سه', 'چند', 'همه', 'باید', 'بود',
            'داد', 'گذاری', 'گیرد', 'بین', 'پس', 'توسط', 'حتی'
        }
        
        filtered_lemmas = [
            (lemma, count) for lemma, count in lemma_freq.most_common(15) 
            if lemma not in stop_words and len(lemma) > 1
        ]
        
        top_lemmas = [l[0] for l in filtered_lemmas[:3]]
        
        if not top_lemmas:
            title = "بدون عنوان"
        elif len(top_lemmas) == 1:
            title = top_lemmas[0]
        else:
            title = ' - '.join(top_lemmas)
        
        return {
            "text": request.text,
            "generated_title": title,
            "top_lemmas": top_lemmas,
            "lemma_frequencies": dict(filtered_lemmas[:5]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title extraction error: {str(e)}")


@app.post("/split/sentences/advanced")
async def split_sentences_advanced(request: SentenceSplitRequest):
    try:
        normalized = normalizer.normalize(request.text)
        
        sentences = sent_tokenize(normalized)
        
        result_sentences = []
        for idx, sent in enumerate(sentences):
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            
            words = word_tokenize(sent_clean)
            
            result_sentences.append({
                "index": idx,
                "text": sent_clean,
                "word_count": len(words),
                "char_count": len(sent_clean),
                "starts_with_capital": sent_clean[0].isupper() if sent_clean else False
            })
        
        return {
            "original": request.text,
            "normalized": normalized,
            "sentences": result_sentences,
            "total_sentences": len(result_sentences)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced sentence splitting error: {str(e)}")

@app.post("/chunk/semantic")
async def semantic_chunk_text(request: SemanticChunkRequest):
    try:
        # Step 1: Normalize text
        normalized = normalizer.normalize(request.text)
        
        if not normalized.strip():
            return {
                "original_text": request.text,
                "normalized_text": normalized,
                "chunks": [],
                "total_chunks": 0,
                "min_chunk_size": request.min_chunk_size,
                "max_chunk_size": request.max_chunk_size,
                "warning": "Empty text after normalization"
            }
        
        # Step 2: Split into paragraphs (improved regex)
        paragraphs = re.split(r'\n\s*\n+', normalized.strip())
        
        # Step 3: Process paragraphs into structured units
        paragraph_units = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            sentences = sent_tokenize(p)
            if not sentences:
                continue
                
            word_count = sum(len(word_tokenize(s)) for s in sentences)
            
            paragraph_units.append({
                "text": p,
                "sentences": sentences,
                "sentence_count": len(sentences),
                "word_count": word_count
            })
        
        if not paragraph_units:
            return {
                "original_text": request.text,
                "normalized_text": normalized,
                "chunks": [],
                "total_chunks": 0,
                "min_chunk_size": request.min_chunk_size,
                "max_chunk_size": request.max_chunk_size,
                "warning": "No valid paragraphs found"
            }
        
        # Step 4: Build chunks intelligently
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        current_para_count = 0
        
        def finalize_chunk():
            """Helper to finalize current chunk"""
            nonlocal current_chunk_sentences, current_word_count, current_para_count
            
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "word_count": current_word_count,
                    "sentence_count": len(current_chunk_sentences),
                    "paragraph_count": current_para_count,
                    "index": len(chunks),
                    "char_count": len(chunk_text)
                })
                current_chunk_sentences = []
                current_word_count = 0
                current_para_count = 0
        
        for para_unit in paragraph_units:
            # Case 1: Single paragraph exceeds max_chunk_size - split by sentences
            if para_unit["word_count"] > request.max_chunk_size:
                # Finalize any pending chunk first
                finalize_chunk()
                
                for sentence in para_unit["sentences"]:
                    sent_words = len(word_tokenize(sentence))
                    
                    # Case 1a: Single sentence exceeds max - force split
                    if sent_words > request.max_chunk_size:
                        if current_chunk_sentences:
                            finalize_chunk()
                        
                        # Split long sentence into words
                        words = word_tokenize(sentence)
                        temp_words = []
                        temp_count = 0
                        
                        for word in words:
                            temp_words.append(word)
                            temp_count += 1
                            
                            if temp_count >= request.max_chunk_size:
                                chunks.append({
                                    "text": " ".join(temp_words),
                                    "word_count": temp_count,
                                    "sentence_count": 1,
                                    "paragraph_count": 1,
                                    "index": len(chunks),
                                    "char_count": len(" ".join(temp_words)),
                                    "is_partial_sentence": True
                                })
                                temp_words = []
                                temp_count = 0
                        
                        if temp_words:
                            current_chunk_sentences = [" ".join(temp_words)]
                            current_word_count = temp_count
                            current_para_count = 1
                    
                    # Case 1b: Adding sentence would exceed max
                    elif current_word_count + sent_words > request.max_chunk_size:
                        finalize_chunk()
                        current_chunk_sentences = [sentence]
                        current_word_count = sent_words
                        current_para_count = 1
                    
                    # Case 1c: Can add sentence to current chunk
                    else:
                        current_chunk_sentences.append(sentence)
                        current_word_count += sent_words
                        if not current_para_count:
                            current_para_count = 1
            
            # Case 2: Adding paragraph would exceed max - finalize and start new
            elif current_word_count + para_unit["word_count"] > request.max_chunk_size:
                # Only finalize if we meet min_chunk_size or no choice
                if current_word_count >= request.min_chunk_size or not current_chunk_sentences:
                    finalize_chunk()
                else:
                    # Try to add at least one sentence from new paragraph
                    finalize_chunk()
                
                current_chunk_sentences = para_unit["sentences"]
                current_word_count = para_unit["word_count"]
                current_para_count = 1
            
            # Case 3: Can add entire paragraph to current chunk
            else:
                current_chunk_sentences.extend(para_unit["sentences"])
                current_word_count += para_unit["word_count"]
                current_para_count += 1
        
        # Finalize last chunk
        finalize_chunk()
        
        # Step 5: Add context windows
        for i, chunk in enumerate(chunks):
            # Previous context
            if i > 0:
                prev_text = chunks[i-1]["text"]
                chunk["prev_context"] = prev_text[-200:] if len(prev_text) > 200 else prev_text
            else:
                chunk["prev_context"] = ""
            
            # Next context
            if i < len(chunks) - 1:
                next_text = chunks[i+1]["text"]
                chunk["next_context"] = next_text[:200] if len(next_text) > 200 else next_text
            else:
                chunk["next_context"] = ""
        
        # Step 6: Calculate statistics
        stats = {
            "total_words": sum(c["word_count"] for c in chunks),
            "avg_chunk_size": sum(c["word_count"] for c in chunks) / len(chunks) if chunks else 0,
            "min_actual_size": min(c["word_count"] for c in chunks) if chunks else 0,
            "max_actual_size": max(c["word_count"] for c in chunks) if chunks else 0,
            "total_sentences": sum(c["sentence_count"] for c in chunks),
            "total_paragraphs": sum(c["paragraph_count"] for c in chunks)
        }
        
        return {
            "original_text": request.text,
            "normalized_text": normalized,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "min_chunk_size": request.min_chunk_size,
            "max_chunk_size": request.max_chunk_size,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Semantic chunking error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Semantic chunking error: {str(e)}")


@app.post("/analyze/text")
async def analyze_text_complete(request: TextRequest):
    try:
        normalized = normalizer.normalize(request.text)
        
        sentences = sent_tokenize(normalized)
        words = word_tokenize(normalized)
        
        unique_words = set(words)
        
        pos_tags = []
        if tagger is not None:
            try:
                pos_tags = tagger.tag(words)
            except:
                pass
        
        lemmas = [lemmatizer.lemmatize(word) for word in words]
        stems = [stemmer.stem(word) for word in words]
        
        return {
            "original": request.text,
            "normalized": normalized,
            "statistics": {
                "char_count": len(normalized),
                "word_count": len(words),
                "unique_word_count": len(unique_words),
                "sentence_count": len(sentences),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0
            },
            "sentences": sentences[:5],
            "words": words[:20],
            "lemmas": lemmas[:20],
            "stems": stems[:20],
            "pos_tags": [{"word": w, "tag": t} for w, t in pos_tags[:20]] if pos_tags else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis error: {str(e)}")


@app.post("/preprocess/document")
async def preprocess_document(request: TextRequest):
    try:
        normalized = normalizer.normalize(request.text)
        
        sentences = sent_tokenize(normalized)
        
        processed_sentences = []
        for sent in sentences:
            words = word_tokenize(sent)
            lemmas = [lemmatizer.lemmatize(w) for w in words]
            
            processed_sentences.append({
                "original": sent,
                "words": words,
                "lemmas": lemmas
            })
        
        all_words = word_tokenize(normalized)
        all_lemmas = [lemmatizer.lemmatize(w) for w in all_words]
        
        return {
            "original": request.text,
            "normalized": normalized,
            "processed_sentences": processed_sentences,
            "all_words": all_words,
            "all_lemmas": all_lemmas,
            "statistics": {
                "sentence_count": len(sentences),
                "word_count": len(all_words),
                "unique_lemmas": len(set(all_lemmas))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document preprocessing error: {str(e)}")


@app.post("/extract/title")
async def extract_title_from_text(request: TextRequest):
    try:
        normalized = normalizer.normalize(request.text)
        words = word_tokenize(normalized)
        
        lemmas = [lemmatizer.lemmatize(w) for w in words if len(w) > 2]
        
        from collections import Counter
        lemma_freq = Counter(lemmas)
        
        stop_words = {'است', 'شد', 'می', 'را', 'به', 'از', 'در', 'که', 'این', 'آن', 'با', 'برای'}
        
        filtered_lemmas = [l for l in lemma_freq.most_common(10) if l[0] not in stop_words]
        
        top_lemmas = [l[0] for l in filtered_lemmas[:3]]
        
        title = ' - '.join(top_lemmas) if top_lemmas else "بدون عنوان"
        
        return {
            "text": request.text,
            "generated_title": title,
            "top_lemmas": top_lemmas,
            "lemma_frequencies": dict(filtered_lemmas[:5])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title extraction error: {str(e)}")
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
