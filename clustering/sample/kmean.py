import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def plot_kmeans_delivery_zones():
    # 1. Generate the customer dataset (3 distinct neighborhoods)
    X_customers, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    
    # 2. Fit K-Means (We want K=3 delivery hubs)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_customers)
    
    # Extract the results
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # 3. Setup the Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 4. Create a grid to map out the "Delivery Zones" (Decision Boundaries)
    # We test thousands of background points to see which hub they are closest to
    h = 0.02  # Step size for the grid
    x_min, x_max = X_customers[:, 0].min() - 1, X_customers[:, 0].max() + 1
    y_min, y_max = X_customers[:, 1].min() - 1, X_customers[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the cluster for every background pixel
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Paint the background zones (Voronoi cells)
    ax.imshow(Z, interpolation='nearest', 
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap='Pastel1', aspect='auto', origin='lower', alpha=0.6)
    
    # 5. Plot the actual customer locations
    ax.scatter(X_customers[:, 0], X_customers[:, 1], c=labels, cmap='Set1', 
               edgecolor='k', s=40, zorder=3, label='Customers')
    
    # 6. Plot the Centroids (The exact Delivery Hub coordinates)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='white', marker='X', 
               s=250, edgecolor='black', linewidth=2, zorder=4, 
               label='Delivery Hubs (Centroids)')
    
    # Formatting
    ax.set_title("K-Means Clustering ($K=3$)\nDelivery Hubs and Zone Boundaries")
    ax.set_xlabel("Map X Coordinate")
    ax.set_ylabel("Map Y Coordinate")
    ax.legend(loc='lower right')
    
    # Force equal proportions so distance is visually accurate
    ax.set_aspect('equal')
    plt.savefig('kmean.jpg')

# Run the visualization
plot_kmeans_delivery_zones()
