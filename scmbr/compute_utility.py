import argparse
import json
import evaluate
import numpy as np
import tqdm
from pathlib import Path

from scmbr.bleurt import BLEURT
from scmbr.bertscore import Bertscore

def load_utility(name):
    if name.startswith("local"):
        name, path = name.split(":")
        name = name.split("-")[1]
        if name == "bertscore":
            return Bertscore(model_name=path)
        elif name == "bleurt":
            return BLEURT(model_name=path)
        else:
            raise ValueError(f"Unknown utility: {name}")

    if name == "bertscore":
        return evaluate.load("bertscore", keep_in_memory=True)
    elif name == "bleurt":
        return evaluate.load("bleurt", "BLEURT-20-D6", keep_in_memory=True)
    else:
        raise ValueError(f"Unknown utility: {name}")

def process_name(name):
    if name.startswith("local"):
        path = name.split(":")[1]
        path = Path(path)
        return path.name
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="communicative-intent")
    parser.add_argument("--split", "-s", type=str, default="test")
    parser.add_argument("--utility", "-u", type=str, default="bleurt", help="comma-separated list")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    args = parser.parse_args()

    batch_size = args.batch_size

    data_dir = Path(__file__).parent / ".." / "dataset"
    data_file = data_dir / f"{args.dataset}.json"
    with open(data_file, "r") as f:
        data = json.load(f)
    data = data["data"]

    # Filter by split
    data = [item for item in data if item["split"] == args.split]

    utility_names = args.utility.split(",")
    utilities = [load_utility(name) for name in utility_names]

    for utility_name, utility in zip(utility_names, utilities):
        print(f"Computing {utility_name} for {args.dataset} ({args.split})...")

        utility_matrix = [] # inputs may have different numbers of generations 
        ncomputations = 0 
        for item in data:
            ncomputations += len(item["generations"]) ** 2
        
        tbar = tqdm.tqdm(total=ncomputations)    
        for item in data:
            item_utility_matrix = []
            generations = item["generations"]
            for i, gen1 in enumerate(generations):
                prediction = gen1
                references = generations
                predictions = [prediction] * len(references)

                if utility_name.startswith("local-"):
                    row = utility.compute(candidates=predictions, references=references)
                elif utility_name == "bertscore":
                    row = utility.compute(predictions=predictions, references=references,
                                          lang="en", model_type="roberta-large", 
                                          batch_size=batch_size)['f1']
                elif utility_name == "bleurt":
                    row = utility.compute(references=references, predictions=predictions)['scores']
                else:
                    raise ValueError(f"Unknown utility: {utility_name}")

                item_utility_matrix.append(row)

                tbar.update(len(generations))
            utility_matrix.append(item_utility_matrix)
        tbar.close()

        utility_matrix = np.array(utility_matrix, dtype=object)            

        utilities_dir = Path(__file__).parent / ".." / "precomputed-utilities"
        utilities_dir.mkdir(exist_ok=True)
        storage_name = process_name(utility_name)
        output_file = utilities_dir / f"{args.dataset}-{args.split}-{storage_name}.npy"
        np.save(output_file, utility_matrix, allow_pickle=True)    