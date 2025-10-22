import numpy as np
from scipy.stats import spearmanr
from scmbr.mbr import target_label_mbr

def compute_structural_optimality(dataset, utilities, mbr_method, method_utilities=None):
    if method_utilities is None:
        method_utilities = utilities
    co = []
    corc = []
    for input_idx in range(len(dataset)):
        utility_matrix = np.array(utilities[input_idx])
        method_utility_matrix = np.array(method_utilities[input_idx])
        gold_labels = dataset[input_idx]["labels"]
        generations = dataset[input_idx]["generations"]

        ranking = mbr_method(method_utility_matrix, generations)
        mbr_label = gold_labels[ranking[0]]

        in_cluster_ranking = target_label_mbr(utility_matrix, gold_labels, 
                                              target_label=mbr_label)

        filtered_ranking = [idx for idx in ranking if idx in in_cluster_ranking]

        spearman_corr, _ = spearmanr(filtered_ranking, in_cluster_ranking)
        corc.append(spearman_corr)

        is_cluster_optimal = generations[ranking[0]] == generations[in_cluster_ranking[0]]
        co.append(is_cluster_optimal)
    return co, corc
