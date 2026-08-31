#!/bin/bash
# End-to-end demo: download the prebuilt mini index, run BM25 + ANCE
# retrieval on TREC DL2019 queries, and evaluate both against qrels.
set -e
cd "$(dirname "$0")/.."

bash data/download_data.sh

python src/retrieve.py --retriever bm25 \
  --index_path data/lucene-index-dl2019-mini \
  --queries_path data/queries_DL2019.jsonl \
  --out_path output/bm25_run.tsv

python src/retrieve.py --retriever ance \
  --index_path data/ance_faiss_dl2019_mini/ance_dl2019_mini_faiss \
  --queries_path data/queries_DL2019.jsonl \
  --out_path output/ance_run.tsv

echo -e "\n=== BM25 ==="
python src/evaluate.py --run_path output/bm25_run.tsv --qrels_path data/qrels_DL2019.jsonl

echo -e "\n=== ANCE ==="
python src/evaluate.py --run_path output/ance_run.tsv --qrels_path data/qrels_DL2019.jsonl
