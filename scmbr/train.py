import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from scmbr.utils import load_structural_variation_dataset, load_utilities
from scmbr.structural_optimality import compute_structural_optimality
from scmbr.mbr import structure_embeddings_mbr, cutoff_mbr
from sentence_transformers import SentenceTransformer
from scmbr.cluster import ClusteringModel

def train_cutoff(data, utilities):
    min_threshold = np.min([np.min(u) for u in utilities["train"]])
    max_threshold = np.max([np.max(u) for u in utilities["train"]])
    thresholds = np.linspace(min_threshold, max_threshold, 50)
    all_train_co = []
    all_train_corc = []
    all_val_co = []
    all_val_corc = []

    for threshold in thresholds:
        mbr_fn = lambda utility_matrix, _: cutoff_mbr(utility_matrix, threshold, c=0.)
        co, corc = compute_structural_optimality(data["train"], utilities["train"], mbr_fn)
        co = np.mean(co)
        corc = np.mean(corc)
        all_train_co.append(co)
        all_train_corc.append(corc)
        val_co, val_corc = compute_structural_optimality(data["validation"], 
                                                      utilities["validation"], mbr_fn)
        val_co = np.mean(val_co)
        val_corc = np.mean(val_corc)
        all_val_co.append(val_co)
        all_val_corc.append(val_corc)

    best_idx = np.argmax(all_train_co)
    print(f"Best cutoff threshold:")
    print(f"Threshold: {thresholds[best_idx]:.3f}, Training CO: {all_train_co[best_idx]:.2f}, Training CORC: {all_train_corc[best_idx]:.2f}, "
          f"Validation CO: {all_val_co[best_idx]:.2f}, Validation CORC: {all_val_corc[best_idx]:.2f}")

def train_clustering(data, dataset_name, learning_rate, run_name):
    model_name = "all-mpnet-base-v2"
    run_name = run_name if run_name else f"{model_name}-{dataset_name}"
    model_dir = Path(__file__).parent / ".." / "models" / "structure-embeddings" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    clustering_model = ClusteringModel(model_name=model_name)
    clustering_model.train(data, model_dir, run_name=run_name, lr=learning_rate)
    print(f"Model saved to {model_dir}")

def train_structure_embeddings(data, utilities, cluster_model="cluster-all-1e-5"):
    cluster_model_dir = Path(__file__).parent / ".." / "models" / "structure-embeddings" / cluster_model
    cluster_model = SentenceTransformer(str(cluster_model_dir))

    thresholds = [None] + np.linspace(0., 1., 50).tolist()
    all_train_co = []
    all_train_corc = []
    all_val_co = []
    all_val_corc = []

    for threshold in thresholds:
        mbr_fn = lambda utility_matrix, generations: structure_embeddings_mbr(
            utility_matrix, generations, cluster_model, cosine_threshold=threshold
        )
        co, corc = compute_structural_optimality(data["train"], utilities["train"], mbr_fn)
        co = np.mean(co)
        corc = np.mean(corc)
        all_train_co.append(co)
        all_train_corc.append(corc)

        val_co, val_corc = compute_structural_optimality(data["validation"], utilities["validation"], mbr_fn)
        val_co = np.mean(val_co)
        val_corc = np.mean(val_corc)
        all_val_co.append(val_co)
        all_val_corc.append(val_corc)

    best_idx = np.argmax(all_train_co)
    thr = thresholds[best_idx]
    thr_str = "None" if thr is None else f"{thr:.3f}"

    print(f"Best cosine threshold:")
    print(f"Threshold: {thr_str}, Training CO: {all_train_co[best_idx]:.2f}, Training CORC: {all_train_corc[best_idx]:.2f}, "
          f"Validation CO: {all_val_co[best_idx]:.2f}, Validation CORC: {all_val_corc[best_idx]:.2f}")

def main():
    valid_methods = ["cluster", "cutoff-bleurt", "cutoff-bertscore", "structure-embeddings-bleurt", "structure-embeddings-bertscore"]
    parser = argparse.ArgumentParser(description="Train clustering model")
    parser.add_argument("--method", "-m", type=str, required=False, default="cluster", 
                        choices=valid_methods)
    parser.add_argument("--dataset", "-d", type=str, required=False, default="all")
    parser.add_argument("--name", "-n", type=str, required=False, default=None)
    parser.add_argument("--learning_rate", "-lr", type=float, required=False, default=5e-5)
    parser.add_argument("--cluster_model", type=str, required=False)
    args = parser.parse_args()

    if args.dataset == "all":
        data = {
            "train": [],
            "validation": [],
            "test": []
        }
        for dataset_name in ["communicative-intent", "emotion", "generation-structure"]:
            data_i, _ = load_structural_variation_dataset(dataset_name)
            data["train"].extend(data_i["train"])
            data["validation"].extend(data_i["validation"])
            data["test"].extend(data_i["test"])
    else:
        data, _ = load_structural_variation_dataset(args.dataset)

    if args.method == "cluster":
        train_clustering(data, args.dataset, args.learning_rate, args.name)
    elif args.method.startswith("cutoff"):
        utility_name = args.method.split("-")[1]
        print(f"utility = {utility_name}")
        print(f"dataset = {args.dataset}")
        utilities = load_utilities(args.dataset, utility_name, sets=["train", "validation"])
        train_cutoff(data, utilities)
    elif args.method.startswith("structure-embeddings"):
        utility_name = args.method.split("-")[-1]
        print(f"utility = {utility_name}")
        print(f"dataset = {args.dataset}")
        utilities = load_utilities(args.dataset, utility_name, sets=["train", "validation"])
        train_structure_embeddings(data, utilities, cluster_model=args.cluster_model)
    else:
        raise ValueError(f"Unknown method: {args.method}.")
