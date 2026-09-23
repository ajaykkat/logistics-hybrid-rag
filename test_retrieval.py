import copy
import io
import json
from pathlib import Path
import unittest
from urllib.parse import urlencode
from retrieval import Engine,load_engine,chunk_document,rrf
from evaluate import metrics
from app import application
from ollama_adapter import OllamaSelector

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=load_engine();cls.docs=json.loads(Path(__file__).with_name('corpus.json').read_text())
    def test_chunks_match_source_offsets(self):
        for chunk in self.engine.chunks:
            self.assertEqual(chunk.text,self.engine.documents[chunk.document_id]['text'][chunk.start:chunk.end])
    def test_long_paragraph_has_no_dropped_words(self):
        doc={**self.docs[0],'text':' '.join(['shipment']*200)}
        chunks=chunk_document(doc,80)
        self.assertEqual(' '.join(c.text for c in chunks),doc['text'])
        self.assertTrue(all(len(c.text)<=80 for c in chunks))
    def test_customer_scope_before_index(self):
        self.assertNotIn('cedar-cold',self.engine.documents)
        self.assertFalse(any('CEDAR-DEMO-ONLY' in c.text for c in self.engine.chunks))
    def test_inactive_excluded(self):self.assertNotIn('delay-old',self.engine.documents)
    def test_current_policy_retrieved(self):
        result=self.engine.answer('When should Northstar escalate a late shipment?')
        self.assertEqual(result['retrieved'][0]['document_id'],'delay')
        self.assertNotIn('240 minutes',result['answer'])
    def test_other_scope_can_access_own_policy(self):
        engine=Engine(self.docs,tenant='cedar')
        self.assertIn('cedar-cold',engine.documents);self.assertNotIn('temperature',engine.documents)
    def test_sparse_exact_identifier(self):
        self.assertEqual(self.engine.search('EDI 214',mode='bm25')[0]['document_id'],'edi')
    def test_all_four_modes(self):
        for mode in ['bm25','dense','hybrid','hybrid-rerank']:
            self.assertTrue(self.engine.search('detention charges',mode))
    def test_unknown_query_abstains(self):
        self.assertEqual(self.engine.answer('chocolate cake recipe')['status'],'abstained')
    def test_low_coverage_abstains(self):
        self.assertEqual(self.engine.answer('salary warehouse manager')['status'],'abstained')
    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):self.engine.search(' ')
    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):self.engine.search('shipment','magic')
    def test_invalid_k_rejected(self):
        with self.assertRaises(ValueError):self.engine.search('shipment',k=0)
    def test_fabricated_quote_rejected(self):
        rows=self.engine.search('detention')
        with self.assertRaises(ValueError):self.engine.verify([{'chunk_id':rows[0]['id'],'quote':'Free loading time is 900 minutes.'}],rows)
    def test_unretrieved_citation_rejected(self):
        rows=self.engine.search('detention',k=1)
        other=next(c for c in self.engine.chunks if c.id!=rows[0]['id'])
        with self.assertRaises(ValueError):self.engine.verify([{'chunk_id':other.id,'quote':other.text}],rows)
    def test_empty_citations_rejected(self):
        with self.assertRaises(ValueError):self.engine.verify([],[])
    def test_extra_claim_field_rejected(self):
        rows=self.engine.search('detention')
        with self.assertRaises(ValueError):self.engine.verify([{'chunk_id':rows[0]['id'],'quote':rows[0]['text'],'claim':'pay immediately'}],rows)
    def test_citation_offsets_verified(self):
        answer=self.engine.answer('What evidence is required for detention charges?')
        self.assertEqual(answer['status'],'supported_quotes')
        for cite in answer['citations']:
            source=self.engine.documents[cite['document_id']]['text']
            self.assertEqual(source[cite['start']:cite['end']],cite['quote'])
    def test_changed_source_rejected(self):
        engine=Engine(copy.deepcopy(self.docs));rows=engine.search('detention')
        doc=rows[0]['document_id'];engine.documents[doc]['text']='modified'
        with self.assertRaises(ValueError):engine.verify([{'chunk_id':rows[0]['id'],'quote':rows[0]['text']}],rows)
    def test_bad_model_output_abstains(self):
        r=self.engine.answer('detention charges',selector=lambda q,c:[{'chunk_id':c[0]['id'],'quote':'invented unsupported claim'}])
        self.assertEqual(r['status'],'abstained')
    def test_rrf_merges_rankings(self):
        values=rrf([['a','b'],['b','c']]);self.assertGreater(values['b'],values['a'])
    def test_metric_deduplicates_documents(self):
        result=metrics(['a','a','b'],['a','b']);self.assertEqual(result['recall_at_5'],1)
    def test_duplicate_documents_rejected(self):
        with self.assertRaises(ValueError):Engine(self.docs+[self.docs[0]])
    def test_api_search(self):
        status=[];body=b''.join(application(self.engine)({'REQUEST_METHOD':'GET','PATH_INFO':'/api/search','QUERY_STRING':urlencode({'q':'detention charges'})},lambda s,h:status.append(s)))
        self.assertEqual(status,['200 OK']);self.assertTrue(json.loads(body)['retrieved'])
    def test_api_invalid_input(self):
        status=[];application(self.engine)({'REQUEST_METHOD':'GET','PATH_INFO':'/api/search'},lambda s,h:status.append(s))
        self.assertEqual(status,['400 Bad Request'])
    def test_ollama_contract_mocked(self):
        rows=self.engine.search('detention');expected=[{'chunk_id':rows[0]['id'],'quote':rows[0]['text']}]
        def transport(request,timeout):
            sent=json.loads(request.data);self.assertFalse(sent['stream']);self.assertEqual(sent['format'],'json')
            return io.BytesIO(json.dumps({'response':json.dumps({'citations':expected})}).encode())
        self.assertEqual(OllamaSelector('mock-model',transport)('detention',rows),expected)
    def test_ollama_malformed_response(self):
        with self.assertRaises(ValueError):OllamaSelector('mock',lambda *a,**kw:io.BytesIO(b'bad'))('query',[])

if __name__=='__main__':unittest.main()
