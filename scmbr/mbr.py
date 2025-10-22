import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
    
def standard_mbr(utility_matrix):
    exp_U = utility_matrix.mean(-1)
    ranking = exp_U.argsort(axis=0)[::-1]
    return ranking

def cutoff_mbr(utility_matrix, threshold, c=0.):
    U = utility_matrix.copy()
    np.fill_diagonal(U, 0.)
    U = np.where(U >= threshold, utility_matrix, c)
    exp_U = U.sum(-1) / (U.shape[-1]-1)
    ranking = exp_U.argsort(axis=0)[::-1]
    return ranking

def target_label_mbr(utility_matrix, labels, target_label):
    ids_for_label = np.where(np.array(labels) == target_label)[0]
    utility_matrix = utility_matrix[np.ix_(ids_for_label, ids_for_label)]
    exp_U = utility_matrix.mean(-1)
    ranking = [ids_for_label[i] for i in np.argsort(exp_U)[::-1]]
    return np.array(ranking)

def determine_k(embeddings, max_k=6, threshold=0.5, tune_threshold=False):
    silhouette_scores = []
    max_k = min(max_k, embeddings.shape[0]-1)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append((k, score, labels))

    # If the silhouette score is below a threshold, we run 
    max_silhouette_score = max(s[1] for s in silhouette_scores)
    
    if tune_threshold:
        return max_silhouette_score

    if max_silhouette_score < threshold: 
        return 1, [0] * len(embeddings) 
    else:
        best_k, _, best_labels = max(silhouette_scores, key=lambda x: x[1])
        return best_k, best_labels

def structure_embeddings_mbr(utility_matrix, generations, cluster_model, cosine_threshold=None):
    embeddings = cluster_model.encode(generations) # already normalized
    cosine_similarity = np.dot(embeddings, embeddings.T)
    cosine_similarity = (cosine_similarity + 1) / 2 
    if cosine_threshold is not None:
        cosine_similarity[cosine_similarity < cosine_threshold] = 0.
    U = cosine_similarity * utility_matrix
    np.fill_diagonal(U, 0.)
    exp_U = U.sum(-1) / (U.shape[-1]-1)
    ranking = exp_U.argsort(axis=0)[::-1]
    return ranking

def cluster_mbr(utility_matrix, generations, cluster_model, nclusters=None, full_ranking=False, threshold=0.9):    
    embeddings = cluster_model.encode(generations)

    # Use given number of clusters if provided, otherwise we need to decide
    if nclusters:
        kmeans = KMeans(n_clusters=nclusters, init='k-means++')
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_
    else:
        nclusters, cluster_labels = determine_k(embeddings, threshold=threshold)

    # Slight efficiency improvement
    if nclusters == 1:
        return standard_mbr(utility_matrix), [0] * len(generations), nclusters
    else:
        if not full_ranking:
            most_common_label = Counter(cluster_labels).most_common(1)[0][0]
            return target_label_mbr(utility_matrix, labels=cluster_labels, target_label=most_common_label), cluster_labels, nclusters
        else:
            ranking = []
            for cluster_label, _ in Counter(cluster_labels).most_common():
                ranking.extend(target_label_mbr(utility_matrix, labels=cluster_labels, target_label=cluster_label))
            return np.array(ranking), cluster_labels, nclusters