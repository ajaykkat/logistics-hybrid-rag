"""Ablate only the literal gate on two synthetic development sets, not held-out data."""
import json
from pathlib import Path
import retrieval
from evaluate import evaluate

def main():
    root=Path(__file__).parent
    engine=retrieval.load_engine()
    guard=retrieval.exact_terms
    report={}
    for name in ['queries.json','queries-challenge.json']:
        queries=json.loads((root/name).read_text())
        try:
            retrieval.exact_terms=lambda query:[]
            baseline=evaluate(engine,queries)
        finally:
            retrieval.exact_terms=guard
        upgraded=evaluate(engine,queries)
        report[name]={'coverage_only':baseline,'literal_gate':upgraded}
        print(name)
        for label,run in report[name].items():print(label,run['summary']['hybrid-rerank'])
    (root/'quality-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
