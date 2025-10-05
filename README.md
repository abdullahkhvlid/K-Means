# KMeans Clustering and Anomaly Detection

## Overview

This project demonstrates the application of **KMeans clustering** and **anomaly detection** on multiple datasets including the **Iris dataset** and synthetic blobs. The goal is to explore how unsupervised learning can identify natural clusters, detect outliers, and reveal hidden patterns in data.

The project showcases real-world and synthetic examples of clustering, cluster visualization, and practical anomaly detection workflows.

## Objectives

* Apply KMeans clustering to real and synthetic datasets
* Evaluate clustering performance using confusion matrices
* Visualize clusters and centroids in 2D space
* Detect anomalies using distances from cluster centroids and standard deviation thresholds
* Strengthen understanding of unsupervised learning and practical data analysis

## Technologies Used

* Python
* Scikit-learn
* NumPy
* Matplotlib
* Seaborn

## Key Steps

1. Load and explore datasets (Iris dataset and synthetic blobs)
2. Apply KMeans clustering with different numbers of clusters
3. Visualize clusters and centroids using scatter plots
4. Generate anomalies in synthetic data and detect them based on distance from centroids
5. Visualize anomalies along with clusters and centroids

## Code Example

```python
from sklearn.datasets import load_iris, make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Iris dataset clustering
iris = load_iris()
X, y = iris.data, iris.target
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.title('KMeans on Iris dataset')
plt.show()

# Synthetic blobs clustering
X_blob, y_blob = make_blobs(n_samples=400, centers=5, cluster_std=1.2, random_state=42)
kmeans_blob = KMeans(n_clusters=5, random_state=42)
kmeans_blob.fit(X_blob)
y_blob_kmeans = kmeans_blob.labels_

plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans_blob.cluster_centers_[:,0], kmeans_blob.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.title('KMeans on synthetic blob dataset')
plt.show()

# Anomaly detection
X_anomaly = np.vstack([X_blob, [[15, 15], [16, 14]]])
kmeans_anom = KMeans(n_clusters=5, random_state=42)
kmeans_anom.fit(X_anomaly)
distances = np.min(np.linalg.norm(X_anomaly[:,None] - kmeans_anom.cluster_centers_, axis=2), axis=1)
threshold = 3 * np.std(distances)
anomalies = X_anomaly[distances > threshold]

plt.scatter(X_anomaly[:,0], X_anomaly[:,1], c='blue', s=50, label='Points')
plt.scatter(anomalies[:,0], anomalies[:,1], c='red', s=100, label='Anomalies')
plt.scatter(kmeans_anom.cluster_centers_[:,0], kmeans_anom.cluster_centers_[:,1], c='black', marker='X', s=200)
plt.legend()
plt.title('Anomaly Detection using KMeans')
plt.show()
```

## Insights

* KMeans clustering effectively identifies natural clusters in both real and synthetic datasets
* Distance-based anomaly detection highlights outliers clearly
* Visualizations demonstrate how clustering and anomaly detection can uncover patterns not immediately visible in raw data

## Next Steps

* Apply KMeans to higher-dimensional datasets and evaluate cluster performance
* Compare KMeans results with other unsupervised algorithms like DBSCAN or Gaussian Mixture Models
* Integrate clustering as preprocessing for supervised learning models

## Author

**Abdullah Muhammad Khalid**
[Portfolio Website](https://abdullahkhalid.vercel.app/)
[GitHub Profile](https://github.com/abdullahkhvlid)

---

