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
        
        normalized = normalizer.normalize(request.text)
        tokens = word_tokenize(normalized)
        tree = chunker.parse(tokens)  # <-- use the instance, not the class
        
        return {
            "text": request.text,
            "normalized": normalized,
            "tokens": tokens,
            "tree": str(tree)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunkinggg error: {str(e)}")

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
        normalized = normalizer.normalize(request.text)
        
        paragraphs = re.split(r'[ \t]*[\n\r]+[ \t]*', normalized)
        
        tokenized_paragraphs = []
        for p in paragraphs:
            if not p.strip():
                continue
            
            sentences = sent_tokenize(p)
            word_count = len(word_tokenize(p))
            
            tokenized_paragraphs.append({
                "text": p,
                "sentences": sentences,
                "word_count": word_count
            })
        
        chunks = []
        current_chunk_text = []
        current_word_count = 0
        
        for para in tokenized_paragraphs:
            if para["word_count"] > request.max_chunk_size:
                if current_chunk_text:
                    chunk_text = "\n".join(current_chunk_text)
                    chunks.append({
                        "text": chunk_text,
                        "word_count": current_word_count,
                        "paragraph_count": len(current_chunk_text),
                        "index": len(chunks)
                    })
                    current_chunk_text = []
                    current_word_count = 0
                
                for sentence in para["sentences"]:
                    sent_words = len(word_tokenize(sentence))
                    
                    if current_word_count + sent_words <= request.max_chunk_size:
                        current_chunk_text.append(sentence)
                        current_word_count += sent_words
                    else:
                        if current_chunk_text:
                            chunk_text = " ".join(current_chunk_text)
                            chunks.append({
                                "text": chunk_text,
                                "word_count": current_word_count,
                                "paragraph_count": 1,
                                "index": len(chunks)
                            })
                        current_chunk_text = [sentence]
                        current_word_count = sent_words
            
            elif current_word_count + para["word_count"] <= request.max_chunk_size:
                current_chunk_text.append(para["text"])
                current_word_count += para["word_count"]
            else:
                if current_chunk_text:
                    chunk_text = "\n".join(current_chunk_text)
                    chunks.append({
                        "text": chunk_text,
                        "word_count": current_word_count,
                        "paragraph_count": len(current_chunk_text),
                        "index": len(chunks)
                    })
                
                current_chunk_text = [para["text"]]
                current_word_count = para["word_count"]
        
        if current_chunk_text:
            chunk_text = "\n".join(current_chunk_text)
            chunks.append({
                "text": chunk_text,
                "word_count": current_word_count,
                "paragraph_count": len(current_chunk_text),
                "index": len(chunks)
            })
        
        for i, chunk in enumerate(chunks):
            chunk["prev_context"] = chunks[i-1]["text"][-200:] if i > 0 else ""
            chunk["next_context"] = chunks[i+1]["text"][:200] if i < len(chunks) - 1 else ""
        
        return {
            "original_text": request.text,
            "normalized_text": normalized,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "min_chunk_size": request.min_chunk_size,
            "max_chunk_size": request.max_chunk_size
        }
    except Exception as e:
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
