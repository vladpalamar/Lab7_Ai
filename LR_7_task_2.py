from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Візуалізація даних
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=None,
    copy_x=True,
)

# Кластеризація даних
kmeans.fit(X)
# Передбачення кластерів
y_kmeans = kmeans.predict(X)
# Візуалізація кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)


# Функція для знаходження кластерів
def find_clusters(X, n_clusters, rseed=2):
    # Рандомізація центрів кластерів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    # Пошук кластерів
    while True:
        # Визначення найближчого центру для кожної точки
        labels = pairwise_distances_argmin(X, centers)

        # Обчислення нових центрів кластерів
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Перевірка на збіжність
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels


# Візуалізація кластерів
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
