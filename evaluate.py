"""Reproduce retrieval ablations on a small synthetic, hand-labeled query set."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import platform
from pathlib import Path
import time
import numpy as np
import sklearn
from retrieval import load_engine

def metrics(ranked,relevant,k=5):
    ranked=list(dict.fromkeys(ranked))[:k]; relevant=set(relevant)
    if not relevant:return None
    hits=[int(doc in relevant) for doc in ranked]
    return {'recall_at_5':sum(hits)/len(relevant),
            'mrr_at_5':next((1/(i+1) for i,v in enumerate(hits) if v),0),
            'ndcg_at_5':sum(h/math.log2(i+2) for i,h in enumerate(hits))/sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))}

def evaluate(engine,queries):
    summary={};details=[]
    for mode in ['bm25','dense','hybrid','hybrid-rerank']:
        values=[];timings=[];unknown=[];answered=[];citation_hits=[]
        for q in queries:
            start=time.perf_counter();result=engine.answer(q['query'],mode);timings.append((time.perf_counter()-start)*1000)
            ranked=[r['document_id'] for r in result['retrieved']]
            measured=metrics(ranked,q['relevant'])
            if measured:
                values.append(measured)
                answered.append(result['status']=='supported_quotes')
                citation_hits.extend(c['document_id'] in q['relevant'] for c in result['citations'])
            else:unknown.append(result['status']=='abstained')
            details.append({'id':q['id'],'mode':mode,'ranked':list(dict.fromkeys(ranked)),
                            'relevant':q['relevant'],'slice':q.get('slice','original'),'metrics':measured,'answer_status':result['status'],
                            'citation_documents':[c['document_id'] for c in result['citations']],
                            'reason':result.get('reason'), 'answerability':result.get('answerability')})
        summary[mode]={key:round(float(np.mean([v[key] for v in values])),4) for key in values[0]}
        summary[mode]['unanswerable_abstention_rate']=round(sum(unknown)/len(unknown),4)
        summary[mode]['answerable_response_rate']=round(sum(answered)/len(answered),4)
        summary[mode]['citation_document_precision']=round(sum(citation_hits)/len(citation_hits),4) if citation_hits else None
        summary[mode]['latency_p50_ms']=round(float(np.percentile(timings,50)),3)
        summary[mode]['latency_p95_ms']=round(float(np.percentile(timings,95)),3)
    return {'scope':engine.tenant,'corpus_sha256':engine.fingerprint,'backend':engine.dense.name,
            'reranker':'cross-encoder' if engine.reranker else 'deterministic-feature-baseline',
            'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
            'neural_environment':{n:version(n) for n in ['torch','sentence-transformers','transformers','huggingface-hub']} if engine.dense.name=='sentence-transformers' or engine.reranker else {},
            'query_count':len(queries),'query_sha256':hashlib.sha256(json.dumps(queries,sort_keys=True).encode()).hexdigest(),'warning':'Synthetic development set, not a held-out or production benchmark. Latency is hardware-dependent and excludes index/model loading.',
            'summary':summary,'details':details}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='evaluation.json');p.add_argument('--queries',default=str(Path(__file__).with_name('queries.json')));p.add_argument('--backend',choices=['lsa','neural'],default='lsa');p.add_argument('--neural-rerank',action='store_true');args=p.parse_args()
    engine=load_engine(backend=args.backend,neural_rerank=args.neural_rerank)
    queries=json.loads(Path(args.queries).read_text())
    report=evaluate(engine,queries);Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
