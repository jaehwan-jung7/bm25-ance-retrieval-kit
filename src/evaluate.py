"""Evaluate a run file against qrels (nDCG, MRR, Recall).

Usage:
    python src/evaluate.py --run_path output/bm25_run.tsv --qrels_path data/qrels_DL2019.jsonl
"""
import argparse
import json
from collections import defaultdict

import pytrec_eval

METRICS = {
    "ndcg_cut_10": "ndcg@10",
    "map_cut_100": "map@100",
    "recall_100": "recall@100",
    "recip_rank": "mrr",
}


def load_run(path):
    run = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, did, _rank, score = line.strip().split("\t")
            run[qid][did] = float(score)
    return run


def load_qrels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # {qid: {did: grade}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--qrels_path", type=str, required=True)
    args = parser.parse_args()

    run = load_run(args.run_path)
    qrels = load_qrels(args.qrels_path)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(METRICS))
    results = evaluator.evaluate(run)

    print(f"Evaluated {len(results)} queries from {args.run_path}\n")
    for measure, label in METRICS.items():
        scores = [q[measure] for q in results.values()]
        overall = sum(scores) / len(scores)
        print(f"{label:12s}: {overall:.4f}")
