import argparse
import numpy as np
from scmbr.utils import load_structural_variation_dataset, load_utilities, select_mbr
from scmbr.structural_optimality import compute_structural_optimality

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "-d", type=str, required=True)
    p.add_argument("--split", "-s", type=str, default="test")
    p.add_argument("--method", "-m", type=str, required=True) 
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cluster_model", type=str, default="all-mpnet-base-v2")
    args = p.parse_args()
    method = "-".join(args.method.split("-")[:-1])
    utility = args.method.split("-")[-1]

    data, metadata = load_structural_variation_dataset(args.dataset)
    data = data[args.split]
    utilities = load_utilities(args.dataset, utility)[args.split]
    mbr_fn = select_mbr(method, args)

    co, corc = compute_structural_optimality(data, utilities, mbr_fn)
    mean_co = float(np.mean(co)) if len(co) else float("nan")
    mean_corc = float(np.nanmean(corc)) if len(corc) else float("nan")
    print(f"split={args.split}")
    print(f"method={method}")
    print(f"utility={utility}")
    print(f"mean_co={mean_co:.4f}")
    print(f"mean_corc={mean_corc:.4f}")