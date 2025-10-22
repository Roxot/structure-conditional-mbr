
import numpy as np

from datasets import Dataset

def construct_triplet_dataset(data, size_per_input=None):
    anchors = []
    positives = []
    negatives = []

    for item in data:
        generations = item["generations"]
        labels = item["labels"]
        anchors_i = []
        positives_i = []
        negatives_i = []

        # If size is set, we will sample anchors randomly
        go_linearly = False
        if size_per_input is None:
            size_per_input = len(generations)
            go_linearly = True

        idx = 0
        while len(anchors_i) < size_per_input:
            anchor_idx = np.random.randint(0, len(generations)) if not go_linearly else idx
            anchor = generations[anchor_idx]
            anchor_label = labels[anchor_idx]

            ids_in_cluster = np.where(np.array(labels) == anchor_label)[0]
            positive = generations[np.random.choice(ids_in_cluster)]

            ids_outside_cluster = np.where(np.array(labels) != anchor_label)[0]
            negative = generations[np.random.choice(ids_outside_cluster)]
            
            anchors_i.append(anchor)
            positives_i.append(positive)
            negatives_i.append(negative)
            idx += 1
        
        anchors.extend(anchors_i)
        positives.extend(positives_i)
        negatives.extend(negatives_i)

    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    })
    return dataset
    