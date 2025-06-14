import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation

url = "https://raw.githubusercontent.com/vianmaulana/Affinity-Propagation/main/synthetic_affinity_dataset.csv"
df = pd.read_csv(url)
X = df[["Feature_1", "Feature_2"]].values

model = AffinityPropagation(random_state=0)
model.fit(X)
labels = model.labels_
centers = model.cluster_centers_indices_
n_clusters = len(np.unique(labels))

plt.figure(figsize=(12, 8))
colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

for cluster_id, color in zip(range(n_clusters), colors):
    cluster_points = X[labels == cluster_id]
    center = X[centers[cluster_id]]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=40, color=color, label=f'Cluster {cluster_id}', alpha=0.6)
    plt.scatter(center[0], center[1], s=200, marker='X', c=[color], edgecolor='k', linewidths=2)

plt.title(f'Affinity Propagation Clustering Result (Total Clusters: {n_clusters})', fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(True)
plt.tight_layout()
plt.show()
