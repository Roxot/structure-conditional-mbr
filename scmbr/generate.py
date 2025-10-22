import argparse
import json
from pathlib import Path
import numpy as np
from scmbr.utils import load_structural_variation_dataset, load_utilities, select_mbr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "-d", type=str, required=True)
    p.add_argument("--split", "-s", type=str, default="test")
    p.add_argument("--method", "-m", type=str, required=True) 
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cluster_model", type=str, default="all-mpnet-base-v2")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    method = "-".join(args.method.split("-")[:-1])
    utility = args.method.split("-")[-1]

    data, metadata = load_structural_variation_dataset(args.dataset)
    data = data[args.split]
    utilities = load_utilities(args.dataset, utility)[args.split]
    mbr_fn = select_mbr(method, args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(data)):
            generations = data[i]["generations"]
            util_matrix = np.array(utilities[i])
            ranking = mbr_fn(util_matrix, generations)
            idx = ranking[0]
            rec = {
                "input_index": data[i]["idx"],
                "method": method,
                "utility": utility,
                "chosen_index": int(idx),
                "chosen_text": generations[int(idx)],
                "ranking": [int(r) for r in ranking],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Generations saved to {out_path}")