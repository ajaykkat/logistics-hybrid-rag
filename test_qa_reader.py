import unittest
from qa_reader import QAReader
from retrieval import load_engine

class ReaderTests(unittest.TestCase):
    def row(self):return {'id':'a','document_id':'first','text':'Do not approve payment. Route this invoice to manual review.'}
    def test_null_abstains(self):
        reader=QAReader(pipeline=lambda **kw:dict(answer='',start=0,end=0,score=.99))
        self.assertEqual(reader.select('Who?', [self.row()])['citations'],[])
    def test_low_score_abstains(self):
        reader=QAReader(pipeline=lambda **kw:dict(answer='payment',start=15,end=22,score=.01))
        self.assertEqual(reader.select('Who?', [self.row()])['citations'],[])
    def test_returned_offsets_must_match_source(self):
        reader=QAReader(pipeline=lambda **kw:dict(answer='approved',start=0,end=8,score=.99))
        self.assertEqual(reader.select('Who?', [self.row()])['citations'],[])
    def test_preserve_negation_and_whole_sentence(self):
        reader=QAReader(pipeline=lambda **kw:dict(answer='payment',start=15,end=22,score=.99))
        self.assertEqual(reader.select('What?', [self.row()])['citations'][0]['quote'],'Do not approve payment.')
    def test_nonfinite_score_rejected(self):
        reader=QAReader(pipeline=lambda **kw:dict(answer='payment',start=15,end=22,score=float('nan')))
        self.assertEqual(reader.select('What?', [self.row()])['citations'],[])
    def test_reader_stays_with_top_ranked_document(self):
        seen=[]
        def predict(**kw):
            seen.append(kw['context'])
            return dict(answer='',start=0,end=0,score=.99)
        reader=QAReader(pipeline=predict)
        reader.select('Question?', [self.row(),{'id':'b','document_id':'other','text':'An unrelated confident passage.'}])
        self.assertEqual(seen,[self.row()['text']])
    def test_invalid_threshold(self):
        for score in [-1,2,float('nan')]:
            with self.assertRaises(ValueError):QAReader(score,pipeline=lambda **kw:None)
    def test_no_reader_call_for_unknown_identifier(self):
        class Forbidden:
            def select(self,*args):raise AssertionError('must not call reader')
        self.assertEqual(load_engine(reader=Forbidden()).answer('What is CEDAR-DEMO-ONLY?')['reason'],'missing_literal')
    def test_null_reader_is_normal_abstention(self):
        class Empty:
            def select(self,*args):return {'citations':[]}
        self.assertEqual(load_engine(reader=Empty()).answer('detention')['reason'],'reader_no_answer')
    def test_reader_cannot_cite_unretrieved_chunk(self):
        class Forged:
            def select(self,*args):return {'citations':[{'chunk_id':'outside','quote':'fabricated response'}]}
        result=load_engine(reader=Forged()).answer('detention')
        self.assertEqual(result['status'],'abstained');self.assertEqual(result['citations'],[])
    def test_incompatible_selectors_rejected(self):
        with self.assertRaises(ValueError):load_engine(reader=object()).answer('detention',selector=lambda *a:[])

if __name__=='__main__':unittest.main()
