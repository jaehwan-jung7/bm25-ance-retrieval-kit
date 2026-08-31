"""Build a BM25 (pyserini/Lucene) or ANCE (FAISS) index from your own corpus.

Only needed if you want to index a different corpus than the bundled DL2019
mini example (see `data/download_data.sh` for the prebuilt indexes).

Corpus format: one JSON object per line, {"id": ..., "contents": ...}
(this is pyserini's "JsonCollection" format - see convert your own corpus to
this format first if needed).

Usage:
    python src/build_index.py --retriever bm25 --corpus_path data/mini_corpus.jsonl \
        --index_path data/lucene-index-mini

    python src/build_index.py --retriever ance --corpus_path data/mini_corpus.jsonl \
        --index_path data/ance_faiss_mini/ance_faiss
"""
import argparse
import json
import os
import subprocess
import tempfile


def build_bm25(corpus_path, index_path, threads=4):
    # pyserini's indexer expects a directory of JsonCollection files.
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.symlink(os.path.abspath(corpus_path), os.path.join(tmp_dir, "docs00.json"))
        subprocess.run(
            [
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", tmp_dir,
                "--index", index_path,
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", str(threads),
                "--storePositions", "--storeDocvectors", "--storeRaw",
            ],
            check=True,
        )


def build_ance(corpus_path, index_path, batch_size=64):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/msmarco-roberta-base-ance-firstp")

    ids, texts = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(int(obj["id"]))
            texts.append(obj["contents"])

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, np.array(ids))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["bm25", "ance"], required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    args = parser.parse_args()

    if args.retriever == "bm25":
        build_bm25(args.corpus_path, args.index_path)
    else:
        build_ance(args.corpus_path, args.index_path)

    print(f"Built {args.retriever} index -> {args.index_path}")
