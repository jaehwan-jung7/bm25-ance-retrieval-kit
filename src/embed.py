"""Extract ANCE embeddings for arbitrary text - useful if you just want the
vectors (e.g. to plug into your own retriever or QPP model), not a run file.

Usage:
    python src/embed.py --input_path my_texts.jsonl --out_path my_embeddings.npy

`my_texts.jsonl` is one JSON object per line: {"id": ..., "text": ...}
Output is a single .npy array of shape [N, 768], row-aligned with the input
order; ids are saved alongside as `<out_path>.ids.json`.
"""
import argparse
import json

import numpy as np


def load_texts(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj["text"])
    return ids, texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    ids, texts = load_texts(args.input_path)
    model = SentenceTransformer("sentence-transformers/msmarco-roberta-base-ance-firstp")
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)

    np.save(args.out_path, embeddings)
    with open(f"{args.out_path}.ids.json", "w") as f:
        json.dump(ids, f)

    print(f"Saved {embeddings.shape} embeddings -> {args.out_path}")
