"""Compare policies using the same neural hybrid retrieval and development queries.

The reader threshold is fixed at 0.25 before this experiment; no tuning loop.
Response coverage and document relevance do not establish answer correctness.
"""
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
from retrieval import load_engine
from qa_reader import QAReader, MODEL, REVISION

def main():
    root=Path(__file__).parent
    engine=load_engine(backend='neural')
    reader=QAReader()
    report={'reader_model':MODEL,'reader_revision':REVISION,'threshold':reader.minimum_score,'backend':'neural','mode':'hybrid',
            'corpus_sha256':engine.fingerprint,
            'environment':{n:version(n) for n in ['torch','transformers','sentence-transformers']},
            'warning':'Synthetic development data. Response coverage is not correctness. No held-out evaluation.',
            'datasets':{}}
    for filename in ['queries.json','queries-challenge.json']:
        questions=json.loads((root/filename).read_text());runs={}
        for name,policy in [('lexical',None),('qa_all_candidates',reader),('qa_top_document',reader)]:
            engine.reader=policy
            reader.candidate_scope='all_candidates' if name=='qa_all_candidates' else 'top_document'
            details=[]
            for q in questions:
                r=engine.answer(q['query'],mode='hybrid')
                details.append({'id':q['id'],'query':q['query'],'relevant':q['relevant'],
                                'status':r['status'],'reason':r.get('reason'),'answer':r['answer'],
                                'citations':r['citations'],'answerability':r['answerability'],
                                'latency_ms':r['latency_ms']})
            known=[r for r in details if r['relevant']];unknown=[r for r in details if not r['relevant']]
            summary={'answerable_responded':sum(r['status']=='supported_quotes' for r in known),'answerable_total':len(known),
                     'unanswerable_rejected':sum(r['status']=='abstained' for r in unknown),'unanswerable_total':len(unknown),
                     'complete_document_coverage':sum(r['status']=='supported_quotes' and set(r['relevant']).issubset({c['document_id'] for c in r['citations']}) for r in known),
                     'wrong_document_responses':sum(r['status']=='supported_quotes' and any(c['document_id'] not in r['relevant'] for c in r['citations']) for r in known)}
            runs[name]={'summary':summary,'details':details};print(filename,name,summary,flush=True)
        report['datasets'][filename]={'query_sha256':hashlib.sha256(json.dumps(questions,sort_keys=True).encode()).hexdigest(),'runs':runs}
    (root/'answerability-comparison.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
