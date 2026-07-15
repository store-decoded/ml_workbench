import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
# agglomorative_hirearchical_clustering
def plot_hierarchical_concept():
    # 1. Generate a small dataset (20 points) so the tree is readable
    np.random.seed(42)
    X_stores, _ = make_blobs(n_samples=20, centers=3, cluster_std=1.2, random_state=42)
    
    # 2. Perform the linkage (the math behind agglomerative clustering)
    # 'ward' linkage minimizes the variance of clusters being merged
    Z = linkage(X_stores, method='ward')
    
    # 3. Create the K=3 labels for the scatter plot to show the final "cut"
    model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = model.fit_predict(X_stores)

    # 4. Setup the Side-by-Side Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot A: The Map (Scatter) ---
    scatter = ax1.scatter(X_stores[:, 0], X_stores[:, 1], c=labels, cmap='viridis', 
                          s=100, edgecolor='k')
    
    # Annotate points with their index so we can track them in the tree
    for i in range(len(X_stores)):
        ax1.annotate(str(i), (X_stores[i, 0], X_stores[i, 1]), 
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
        
    ax1.set_title("Store Locations (Map view)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot B: The Tree (Dendrogram) ---
    # We draw a horizontal line to show where we "cut" the tree to get 3 clusters
    ax2.set_title("The Hierarchy (Dendrogram)")
    dendro = dendrogram(Z, ax=ax2, color_threshold=8) 
    
    # Add a dashed line showing the threshold cut for K=3
    ax2.axhline(y=8, color='r', linestyle='--', label='Cut (yields 3 clusters)')
    ax2.set_xlabel("Store Index (Data Point ID)")
    ax2.set_ylabel("Merge Distance")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('clustering/ahc.jpg')

# Run the visualization
plot_hierarchical_concept()
