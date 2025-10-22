import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from scmbr.mbr import cutoff_mbr, cluster_mbr, structure_embeddings_mbr

def select_mbr(method, args):
    m = method.lower()

    if m == "cutoff":
        def mbr_fn(utility_matrix, *_):
            return cutoff_mbr(
                utility_matrix,
                threshold=args.threshold,
                c=0.
            )
        return mbr_fn

    if m == "cluster":
        cluster_model_dir = Path(__file__).parent / ".." / "models" / "structure-embeddings" / args.cluster_model
        cluster_model = SentenceTransformer(str(cluster_model_dir))
        def mbr_fn(utility_matrix, generations, *_):
            ranking, labels, nclusters = cluster_mbr(
                utility_matrix,
                generations,
                cluster_model,
                nclusters=None,
                full_ranking=True,
            )
            return ranking
        return mbr_fn

    if m in ["structure-embeddings", "structure_embeddings"]:
        cluster_model_dir = Path(__file__).parent / ".." / "models" / "structure-embeddings" / args.cluster_model
        cluster_model = SentenceTransformer(str(cluster_model_dir))
        def mbr_fn(utility_matrix, generations, *_):
            return structure_embeddings_mbr(
                utility_matrix,
                generations,
                cluster_model,
                cosine_threshold=args.threshold
            )
        return mbr_fn

    raise ValueError(f"Unknown method: {method}")

def load_utilities(dataset_name, utility, model_name=None, sets=["train", "validation", "test"]):
    data_dir = Path(__file__).parent / ".." / "precomputed-utilities"
    utilities = {}

    if dataset_name == "all":
        datasets = ["dialogue-act", "emotion", "response-structure"]
    else:
        datasets = [dataset_name]

    for dataset in datasets:
        for set_name in sets:
            model_postfix = "" if model_name is None else f"-{model_name}"
            filename = f"{dataset}-{set_name}-{utility}{model_postfix}.npy"

            U = np.load(data_dir / filename, allow_pickle=True) # [inputs, N, N]
            U = [U[i] for i in range(U.shape[0])]
            if set_name not in utilities:
                utilities[set_name] = U
            else:
                utilities[set_name].extend(U)
            
    return utilities 

def load_structural_variation_dataset(dataset_name):
    data_dir = Path(__file__).parent / ".." / "dataset"

    if dataset_name == "all":
        data = {
            "train": [],
            "validation": [],
            "test": []
        }
        for dataset_name in ["dialogue-act", "emotion", "response-structure"]:
            with open(data_dir / f"{dataset_name}.json") as f:
                data_i = json.load(f)["data"]
                dataset_i = {}
                for split in ["train", "validation", "test"]:
                    split_data = [d for d in data_i if d["split"] == split]
                    dataset_i[split] = split_data

            data["train"].extend(dataset_i["train"])
            data["validation"].extend(dataset_i["validation"])
            data["test"].extend(dataset_i["test"])
        metadata = None
        return data, metadata

    with open(data_dir / f"{dataset_name}.json") as f:
        data = json.load(f)
    metadata = data["metadata"]
    data = data["data"]

    dataset = {}
    for split in ["train", "validation", "test"]:
        split_data = [d for d in data if d["split"] == split]
        dataset[split] = split_data

    return dataset, metadata