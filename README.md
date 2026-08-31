# BM25 & ANCE Retrieval Kit

A minimal, ready-to-run toolkit for two classic first-stage retrievers over
MS MARCO passages: **BM25** (sparse, via [Pyserini](https://github.com/castorini/pyserini))
and **ANCE** (dense, via [sentence-transformers](https://www.sbert.net/)). It
comes with a small prebuilt example index so you can run real retrieval and
evaluation in minutes, plus scripts to build your own index, extract
embeddings, and run retrieval at larger scale.

Contents
- [Environment Setup](#environment-setup)
- [Quickstart](#quickstart)
- [Retrieval](#retrieval)
  - [BM25 (sparse)](#bm25-sparse)
  - [ANCE (dense)](#ance-dense)
- [Extracting Embeddings](#extracting-embeddings)
- [Building Your Own Index](#building-your-own-index)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

## Environment Setup
Pyserini needs a JDK (11+).
```bash
# Debian/Ubuntu, if you don't already have a JDK:
sudo apt-get install -y default-jdk

pip install -r requirements.txt
```

## Quickstart
Downloads a prebuilt BM25 + ANCE index over a ~9K-passage MS MARCO subset
(all passages judged in TREC DL2019), runs both retrievers on the DL2019
queries, and evaluates against qrels - no full 8.8M-passage indexing needed:
```bash
bash examples/quickstart.sh
```
Expect nDCG@10 around 0.46 (BM25) and 0.70 (ANCE). Note the mini corpus only
contains passages judged in TREC DL2019 (no hard negatives), so these
numbers are for a quick sanity check, not comparable to full-corpus
MS MARCO leaderboards.

## Retrieval
### BM25 (sparse)
```bash
python src/retrieve.py --retriever bm25 \
  --index_path data/lucene-index-dl2019-mini \
  --queries_path data/queries_DL2019.jsonl \
  --out_path output/bm25_run.tsv
```

### ANCE (dense)
```bash
python src/retrieve.py --retriever ance \
  --index_path data/ance_faiss_dl2019_mini/ance_dl2019_mini_faiss \
  --queries_path data/queries_DL2019.jsonl \
  --out_path output/ance_run.tsv
```
`--queries_path` takes a JSON file of `{qid: query text}`; the run file is
written as tab-separated `qid, docid, rank, score`.

## Extracting Embeddings
If you just want ANCE embeddings for your own text (not a full retrieval
run), see `src/embed.py`:
```bash
python src/embed.py --input_path my_texts.jsonl --out_path my_embeddings.npy
```

## Building Your Own Index
To index a different corpus (instead of the bundled DL2019 mini example):
```bash
python src/build_index.py --retriever bm25 --corpus_path data/mini_corpus.jsonl \
  --index_path data/lucene-index-mini

python src/build_index.py --retriever ance --corpus_path data/mini_corpus.jsonl \
  --index_path data/ance_faiss_mini/ance_faiss
```
`--corpus_path` takes a JSONL file of `{"id": ..., "contents": ...}` per
line (Pyserini's `JsonCollection` format). For the full MS MARCO passage
corpus and other retrieval datasets, see
[Neural-IR](https://github.com/jaehwan-jung7/Neural-IR).

## Evaluation
```bash
python src/evaluate.py --run_path output/bm25_run.tsv --qrels_path data/qrels_DL2019.jsonl
```
Reports nDCG@10, MAP@100, Recall@100, and MRR.

## Acknowledgement
This repository was developed with support from the **데이터사이언스 융합인재양성사업단**
(Data Science-based Convergent Talent Education Program) - http://dsplus.uos.ac.kr/
