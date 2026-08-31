"""Run BM25 (sparse) or ANCE (dense) retrieval against a prebuilt index.

Usage:
    python src/retrieve.py --retriever bm25 --index_path data/lucene-index-dl2019-mini \
        --queries_path data/queries_DL2019.jsonl --out_path output/bm25_run.tsv

    python src/retrieve.py --retriever ance --index_path data/ance_faiss_dl2019_mini/ance_dl2019_mini_faiss \
        --queries_path data/queries_DL2019.jsonl --out_path output/ance_run.tsv
"""
import argparse
import json
import os

from tqdm import tqdm


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # {qid: text}


def write_run(run: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, did_score in run.items():
            for rank, (did, score) in enumerate(
                sorted(did_score.items(), key=lambda x: x[1], reverse=True), start=1
            ):
                f.write(f"{qid}\t{did}\t{rank}\t{score}\n")


def retrieve_bm25(index_path, queries, top_k):
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index_path)
    run = {}
    for qid, text in tqdm(queries.items(), desc="BM25 retrieval"):
        hits = searcher.search(text, k=top_k)
        run[qid] = {hit.docid: hit.score for hit in hits}
    return run


def retrieve_ance(index_path, queries, top_k, batch_size=32):
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/msmarco-roberta-base-ance-firstp")
    index = faiss.read_index(index_path)

    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    scores, doc_ids = index.search(embeddings, top_k)

    run = {}
    for qid, score_row, docid_row in zip(qids, scores, doc_ids):
        run[qid] = {str(did): float(score) for did, score in zip(docid_row, score_row) if did != -1}
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["bm25", "ance"], required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--queries_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    queries = load_queries(args.queries_path)

    if args.retriever == "bm25":
        run = retrieve_bm25(args.index_path, queries, args.top_k)
    else:
        run = retrieve_ance(args.index_path, queries, args.top_k)

    write_run(run, args.out_path)
    print(f"Wrote {sum(len(v) for v in run.values())} scored (query, doc) pairs -> {args.out_path}")
