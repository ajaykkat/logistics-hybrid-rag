"""Hybrid retrieval with traceable quotes. Corpus files are trusted demo configuration."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

STOP = set(ENGLISH_STOP_WORDS) | {'does', 'did', 'should', 'need', 'please', 'tell'}

def tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.casefold()) if t not in STOP]

@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    tenant: str
    version: str
    start: int
    end: int
    text: str
    sha256: str


def chunk_document(doc, max_chars=700):
    """Paragraph-first chunks; long paragraphs split at whitespace; exact source offsets."""
    if max_chars < 30:
        raise ValueError('max_chars must be at least 30')
    text = doc['text']
    chunks = []
    for match in re.finditer(r'\S(?:.*?\S)?(?=\n\s*\n|\Z)', text, re.S):
        start, stop = match.span()
        while start < stop:
            end = min(start + max_chars, stop)
            if end < stop:
                split = text.rfind(' ', start, end)
                if split > start:
                    end = split
            value = text[start:end]
            digest = hashlib.sha256(value.encode()).hexdigest()
            chunks.append(Chunk(f"{doc['id']}:{start}-{end}", doc['id'], doc['title'], doc['tenant'],
                                doc['version'], start, end, value, digest))
            start = end
            while start < stop and text[start].isspace():
                start += 1
    return chunks


class BM25:
    def __init__(self, texts, k1=1.5, b=.75):
        self.counts = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(texts))
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1+(len(texts)-n+.5)/(n+.5)) for t,n in df.items()}
        self.k1,self.b = k1,b

    def score(self, query):
        values = []
        for counts,length in zip(self.counts,self.lengths):
            total = 0.
            for token in set(tokens(query)):
                f = counts[token]
                if f:
                    total += self.idf.get(token,0)*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*length/(self.average or 1)))
            values.append(total)
        return np.array(values)


class LSAVectors:
    """Offline dense baseline, not a pretrained neural embedding model."""
    name = 'lsa-tfidf'
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, stop_words='english')
        sparse = self.vectorizer.fit_transform(texts)
        dims = min(12, sparse.shape[0]-1, sparse.shape[1]-1)
        self.reducer = TruncatedSVD(n_components=max(1,dims), random_state=42)
        self.matrix = normalize(self.reducer.fit_transform(sparse))

    def score(self, query):
        vector = normalize(self.reducer.transform(self.vectorizer.transform([query])))
        return (self.matrix @ vector.T).ravel()


class NeuralVectors:
    """Optional Sentence Transformers adapter; models download only when explicitly selected."""
    name = 'sentence-transformers'
    def __init__(self, texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode(texts, normalize_embeddings=True)

    def score(self, query):
        return np.asarray(self.matrix @ self.model.encode(query, normalize_embeddings=True)).ravel()


def rrf(rankings, constant=60):
    scores = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking,1):
            scores[item] += 1/(constant+rank)
    return scores


class Engine:
    def __init__(self, documents, tenant='northstar', backend='lsa', neural_rerank=False):
        if backend not in {'lsa','neural'}:
            raise ValueError('backend must be lsa or neural')
        self.tenant = tenant
        self.documents = {}
        seen = set()
        for doc in documents:
            for field in ['id','title','tenant','version','text']:
                if not isinstance(doc.get(field),str) or not doc[field].strip():
                    raise ValueError(f'document requires {field}')
            if doc['id'] in seen:
                raise ValueError('duplicate document IDs')
            seen.add(doc['id'])
            # Scope before indexing: excluded content cannot influence this tenant's vector space.
            if doc.get('active') is True and doc['tenant'] in {tenant,'public'}:
                self.documents[doc['id']] = doc
        self.chunks = [c for doc in self.documents.values() for c in chunk_document(doc)]
        if len(self.chunks) < 2:
            raise ValueError('scope must contain at least two chunks')
        self.by_id = {c.id:c for c in self.chunks}
        self.texts = [c.title+'\n'+c.text for c in self.chunks]
        self.sparse = BM25(self.texts)
        self.dense = LSAVectors(self.texts) if backend=='lsa' else NeuralVectors(self.texts)
        self.reranker = None
        if neural_rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
        self.fingerprint = hashlib.sha256(json.dumps(list(self.documents.values()),sort_keys=True).encode()).hexdigest()

    def search(self, query, mode='hybrid-rerank', k=5):
        if not isinstance(query,str) or not query.strip() or len(query)>2000:
            raise ValueError('query must contain 1 to 2000 characters')
        if mode not in {'bm25','dense','hybrid','hybrid-rerank'} or type(k) is not int or not 1<=k<=20:
            raise ValueError('invalid search mode or k')
        sparse,dense = self.sparse.score(query),self.dense.score(query)
        n=min(20,len(self.chunks))
        sparse_rank=sorted((i for i in range(len(sparse)) if sparse[i]>0),key=lambda i:(-sparse[i],self.chunks[i].id))[:n]
        dense_rank=sorted((i for i in range(len(dense)) if dense[i]>0),key=lambda i:(-dense[i],self.chunks[i].id))[:n]
        fusion=rrf([sparse_rank,dense_rank])
        if mode=='bm25': order=sparse_rank; values={i:float(sparse[i]) for i in order}
        elif mode=='dense': order=dense_rank; values={i:float(dense[i]) for i in order}
        else: order=sorted(fusion,key=lambda i:(-fusion[i],self.chunks[i].id)); values=dict(fusion)
        query_tokens=set(tokens(query))
        coverage={i:len(query_tokens & set(tokens(self.texts[i])))/max(1,len(query_tokens)) for i in order}
        if mode=='hybrid-rerank':
            if self.reranker:
                scores=self.reranker.predict([(query,self.texts[i]) for i in order]) if order else []
                values={i:float(s) for i,s in zip(order,scores)}
            else:
                # Transparent baseline reranker: lexical coverage, title match, dense similarity, RRF.
                values={i: .50*coverage[i]+.20*len(query_tokens & set(tokens(self.chunks[i].title)))/max(1,len(query_tokens))
                           +.20*max(0,float(dense[i]))+.10*fusion[i]*30.5 for i in order}
            order=sorted(order,key=lambda i:(-values[i],self.chunks[i].id))
        return [{**asdict(self.chunks[i]),'score':round(values[i],6),'bm25':round(float(sparse[i]),6),
                 'dense':round(float(dense[i]),6),'rrf':round(fusion.get(i,0),6),
                 'coverage':round(coverage[i],6)} for i in order[:k]]

    def verify(self, citations, retrieved):
        if not isinstance(citations,list) or not 1<=len(citations)<=3:
            raise ValueError('expected one to three citations')
        allowed={r['id'] for r in retrieved}
        checked=[]
        for value in citations:
            if not isinstance(value,dict) or set(value)!={'chunk_id','quote'}:
                raise ValueError('citation must contain only chunk_id and quote')
            cid,quote=value['chunk_id'],value['quote']
            if not isinstance(cid,str) or cid not in allowed or cid not in self.by_id:
                raise ValueError('citation is not from retrieved scope')
            chunk=self.by_id[cid]
            if not isinstance(quote,str) or len(quote.strip())<12 or quote not in chunk.text:
                raise ValueError('quote must be a nontrivial exact source substring')
            source=self.documents[chunk.document_id]['text']
            if hashlib.sha256(source[chunk.start:chunk.end].encode()).hexdigest()!=chunk.sha256:
                raise ValueError('source changed after indexing')
            offset=chunk.start+chunk.text.index(quote)
            checked.append({'chunk_id':cid,'quote':quote,'document_id':chunk.document_id,'title':chunk.title,
                            'version':chunk.version,'start':offset,'end':offset+len(quote),'sha256':chunk.sha256})
        return checked

    def answer(self, query, mode='hybrid-rerank', selector:Callable|None=None):
        started=time.perf_counter()
        retrieved=self.search(query,mode,k=5)
        eligible=[r for r in retrieved if r['coverage']>=.40 and r['bm25']>0]
        result={'query':query,'tenant':self.tenant,'mode':mode,'backend':self.dense.name,
                'reranker':'cross-encoder' if self.reranker else 'deterministic-feature-baseline',
                'corpus_sha256':self.fingerprint,'retrieved':retrieved,'citations':[],
                'generator':'model-evidence-selection' if selector else 'extractive-no-llm'}
        if not eligible:
            result.update(status='abstained',answer='No sufficiently supported evidence found in this scope.')
        else:
            try:
                if selector:
                    proposed=selector(query,eligible)
                else:
                    proposed=[]
                    used=set()
                    for row in eligible:
                        if row['document_id'] in used:continue
                        used.add(row['document_id'])
                        sentences=re.findall(r'[^.!?]+[.!?]?',row['text'])
                        ranked=sorted((s.strip() for s in sentences if len(s.strip())>=12),
                                      key=lambda s:-len(set(tokens(s))&set(tokens(query))))
                        if ranked:proposed.append({'chunk_id':row['id'],'quote':ranked[0]})
                        if len(proposed)==2:break
                citations=self.verify(proposed,eligible)
                result.update(status='supported_quotes',citations=citations,
                              answer='\n\n'.join(f'[{i}] {c["quote"]}' for i,c in enumerate(citations,1)))
            except (ValueError,TypeError,KeyError) as exc:
                result.update(status='abstained',answer='Evidence selection failed citation validation.',error=str(exc))
        result['latency_ms']=round((time.perf_counter()-started)*1000,3)
        return result


def load_engine(path=None, **kwargs):
    source=Path(path) if path else Path(__file__).with_name('corpus.json')
    return Engine(json.loads(source.read_text()),**kwargs)
