"""Local logistics evidence workbench; no network model calls by default."""
import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server
from retrieval import load_engine
from ollama_adapter import OllamaSelector

def application(engine,selector=None):
    def app(env,start_response):
        path=env.get('PATH_INFO','/')
        status='200 OK';kind='application/json; charset=utf-8'
        try:
            if env.get('REQUEST_METHOD')!='GET':
                status='405 Method Not Allowed';data={'error':'GET required'}
            elif path=='/':
                body=Path(__file__).with_name('index.html').read_bytes()
                start_response(status,[('Content-Type','text/html; charset=utf-8'),('X-Content-Type-Options','nosniff')]);return [body]
            elif path=='/api/search':
                params=parse_qs(env.get('QUERY_STRING',''))
                data=engine.answer(params.get('q',[''])[0],params.get('mode',['hybrid-rerank'])[0],selector)
            elif path=='/api/info':
                data={'tenant':engine.tenant,'documents':len(engine.documents),'chunks':len(engine.chunks),
                      'backend':engine.dense.name,'generator':'local-llm' if selector else 'extractive',
                      'corpus_sha256':engine.fingerprint}
            else:status='404 Not Found';data={'error':'not found'}
        except ValueError as exc:status='400 Bad Request';data={'error':str(exc)}
        body=json.dumps(data,allow_nan=False).encode()
        start_response(status,[('Content-Type',kind),('Cache-Control','no-store'),('X-Content-Type-Options','nosniff')]);return [body]
    return app

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tenant',choices=['northstar','cedar'],default='northstar')
    p.add_argument('--backend',choices=['lsa','neural'],default='lsa')
    p.add_argument('--neural-rerank',action='store_true')
    p.add_argument('--ollama-model')
    p.add_argument('--port',type=int,default=8082)
    p.add_argument('--query');p.add_argument('--mode',default='hybrid-rerank',choices=['bm25','dense','hybrid','hybrid-rerank'])
    args=p.parse_args()
    engine=load_engine(tenant=args.tenant,backend=args.backend,neural_rerank=args.neural_rerank)
    selector=OllamaSelector(args.ollama_model) if args.ollama_model else None
    if args.query:
        print(json.dumps(engine.answer(args.query,args.mode,selector),indent=2));return
    print(f'Local synthetic-data demo: http://127.0.0.1:{args.port}')
    with make_server('127.0.0.1',args.port,application(engine,selector)) as server:server.serve_forever()

if __name__=='__main__':main()
