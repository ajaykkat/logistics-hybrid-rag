"""Reproduce retrieval ablations on a small synthetic, hand-labeled query set."""
import argparse
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
        values=[];timings=[];unknown=[]
        for q in queries:
            start=time.perf_counter();result=engine.answer(q['query'],mode);timings.append((time.perf_counter()-start)*1000)
            ranked=[r['document_id'] for r in result['retrieved']]
            measured=metrics(ranked,q['relevant'])
            if measured:values.append(measured)
            else:unknown.append(result['status']=='abstained')
            details.append({'id':q['id'],'mode':mode,'ranked':list(dict.fromkeys(ranked)),
                            'relevant':q['relevant'],'metrics':measured,'answer_status':result['status']})
        summary[mode]={key:round(float(np.mean([v[key] for v in values])),4) for key in values[0]}
        summary[mode]['unanswerable_abstention_rate']=round(sum(unknown)/len(unknown),4)
        summary[mode]['latency_p50_ms']=round(float(np.percentile(timings,50)),3)
        summary[mode]['latency_p95_ms']=round(float(np.percentile(timings,95)),3)
    return {'scope':engine.tenant,'corpus_sha256':engine.fingerprint,'backend':engine.dense.name,
            'reranker':'cross-encoder' if engine.reranker else 'deterministic-feature-baseline',
            'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
            'query_count':len(queries),'warning':'Synthetic development set, not a held-out or production benchmark. Latency is hardware-dependent and excludes index/model loading.',
            'summary':summary,'details':details}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='evaluation.json');p.add_argument('--backend',choices=['lsa','neural'],default='lsa');p.add_argument('--neural-rerank',action='store_true');args=p.parse_args()
    engine=load_engine(backend=args.backend,neural_rerank=args.neural_rerank)
    queries=json.loads(Path(__file__).with_name('queries.json').read_text())
    report=evaluate(engine,queries);Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
