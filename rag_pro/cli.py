from __future__ import annotations

import argparse
import json

from .config import Settings
from .pipeline import RAGPipeline


def main() -> None:
    p = argparse.ArgumentParser(prog="rag-pro", description="Advanced RAG toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="index files/folders (.md .txt .pdf)")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--index", default=".index")

    q = sub.add_parser("query", help="ask a question")
    q.add_argument("question")
    q.add_argument("--index", default=".index")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    q.add_argument("--json", action="store_true")

    sv = sub.add_parser("serve", help="run the FastAPI server")
    sv.add_argument("--index", default=".index")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args()

    if args.cmd == "ingest":
        pipe = RAGPipeline(Settings.from_env())
        stats = pipe.ingest_paths(args.paths)
        pipe.save(args.index)
        print(f"Indexed {stats['documents']} documents / {stats['chunks']} chunks -> {args.index}")
    elif args.cmd == "query":
        pipe = RAGPipeline.load(args.index)
        a = pipe.query(args.question, k=args.k, mode=args.mode)
        if args.json:
            print(json.dumps(a.to_dict(), indent=2))
            return
        print(f"\n{a.answer}\n")
        print(f"confidence={a.confidence}  grounded={a.grounded}  abstained={a.abstained}  timings_ms={a.timings_ms}")
        for c in a.citations:
            print(f"  [{c.n}] {c.doc_id} :: {c.heading}")
    elif args.cmd == "serve":
        import os
        import uvicorn
        os.environ["RAG_INDEX_DIR"] = args.index
        uvicorn.run("rag_pro.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
