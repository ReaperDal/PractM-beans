import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


data = pd.read_csv("customers_features.csv", header=None, names=["age", "income", "spending_score"])
X = data.to_numpy(dtype=float)


X = normalize(X)

def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def kmeans_cosine(X, k, max_iter=100):
    n_samples = X.shape[0]

    if k > n_samples:
        raise ValueError(f"k = {k} більше ніж кількість зразків ({n_samples})")

    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        distances = np.array([[cosine_distance(x, centroid) for centroid in centroids] for x in X])
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            normalize(np.mean(X[labels == j], axis=0).reshape(1, -1))[0]
            if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])


        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


inertias = []
K = range(1, len(X) + 1)

for k in K:
    labels, centroids = kmeans_cosine(X, k)

    inertia = sum(
        cosine_distance(X[i], centroids[labels[i]]) for i in range(len(X))
    )
    inertias.append(inertia)


plt.plot(K, inertias, marker='o')
plt.xlabel("Кількість кластерів (k)")
plt.ylabel("Сумарна косинусна відстань (Inertia)")
plt.title("Elbow-графік для K-Means (косинусна відстань)")
plt.grid(True)
plt.show()
