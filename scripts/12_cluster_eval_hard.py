import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from collections import Counter

Z = np.load("data/features/mm_cvae_latents.npy")
meta = pd.read_csv("data/features/mm_cvae_latents_meta.csv")

labels_true = meta.genre.astype("category").cat.codes.values
labels_pred = KMeans(n_clusters=5, random_state=42).fit_predict(Z)

def purity(y_true, y_pred):
    total = 0
    for c in set(y_pred):
        idx = np.where(y_pred==c)[0]
        total += Counter(y_true[idx]).most_common(1)[0][1]
    return total/len(y_true)

print("Silhouette:", silhouette_score(Z, labels_pred))
print("NMI:", normalized_mutual_info_score(labels_true, labels_pred))
print("ARI:", adjusted_rand_score(labels_true, labels_pred))
print("Purity:", purity(labels_true, labels_pred))
