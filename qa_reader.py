"""Optional local extractive reader with an explicit no-answer option.

Model scores are uncalibrated ranking scores, not probabilities of correctness.
The default 0.25 minimum score is an initial policy, fixed before evaluation.
"""
import math
import re

MODEL = 'deepset/minilm-uncased-squad2'
REVISION = '934656cdda79824eabf503ed56e15c01ddbdbe3f'

class QAReader:
    def __init__(self, minimum_score=0.25, pipeline=None, candidate_scope="top_document"):
        if candidate_scope not in {"top_document","all_candidates"}:
            raise ValueError("invalid candidate scope")
        self.candidate_scope=candidate_scope
        if not isinstance(minimum_score,(int,float)) or not math.isfinite(minimum_score) or not 0<=minimum_score<=1:
            raise ValueError('minimum_score must be finite and between zero and one')
        self.minimum_score=minimum_score
        if pipeline is None:
            from transformers import pipeline as make_pipeline
            pipeline=make_pipeline('question-answering',model=MODEL,tokenizer=MODEL,revision=REVISION,device=-1)
        self.pipeline=pipeline

    def select(self, question, rows):
        candidates=[];trace=[]
        # Preserve retrieval authority: QA scores are not comparable relevance scores.
        if rows and self.candidate_scope=='top_document':
            rows=[r for r in rows if r['document_id']==rows[0]['document_id']]
        for row in rows:
            prediction=self.pipeline(question=question,context=row['text'],handle_impossible_answer=True)
            score=float(prediction['score'])
            start,end=prediction['start'],prediction['end']
            answer=prediction['answer']
            valid=(type(start) is int and type(end) is int and 0<=start<end<=len(row['text'])
                   and row['text'][start:end]==answer and math.isfinite(score))
            trace.append({'chunk_id':row['id'],'score':score,'span':answer,'valid_span':valid})
            if not valid or score<self.minimum_score:continue
            # Return the complete source sentence around the span, preserving context/negation.
            sentences=list(re.finditer(r'[^.!?]+[.!?]?',row['text']))
            covering=[s for s in sentences if s.start()<end and s.end()>start]
            if not covering:continue
            quote=row['text'][covering[0].start():covering[-1].end()].strip()
            if len(quote)<12:continue
            candidates.append((score,row['id'],quote))
        candidates.sort(key=lambda c:(-c[0],c[1]))
        citations=[{'chunk_id':cid,'quote':quote} for _,cid,quote in candidates[:1]]
        return {'citations':citations,'trace':trace,'minimum_score':self.minimum_score,
                'candidate_scope':self.candidate_scope,'policy':'extractive-qa-with-null','model':MODEL,'revision':REVISION}
