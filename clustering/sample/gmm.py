import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
# Gaussian Mixture Models (GMM)
def plot_gmm_ellipses(X,y_true):

    # 2. Stretch the blobs to make them elliptical
    # We use a transformation matrix to warp the standard X, Y coordinates
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_stretched = np.dot(X, transformation)
    
    # 3. Fit the Gaussian Mixture Model
    # 'full' covariance allows each cluster to have its own unique elliptical shape
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_stretched)
    
    # 4. Setup the Plot
    plt.figure(figsize=(10, 6))
    
    # Scatter the points colored by their GMM assigned cluster
    plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels, cmap='viridis', 
                s=30, edgecolor='k', zorder=2)
    
    # 5. Draw the Probability Density Contours (The Gaussian "Bells")
    # Create a grid of points covering the plot area
    x_min, x_max = X_stretched[:, 0].min() - 1, X_stretched[:, 0].max() + 1
    y_min, y_max = X_stretched[:, 1].min() - 1, X_stretched[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Evaluate the GMM model on every point in the grid
    Z = gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Draw contour lines to represent the 2D Gaussian distributions
    plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), 
                cmap='Greys', alpha=0.5, zorder=1)
    
    # Plot the centers (Means)
    centers = gmm.means_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=150, 
                edgecolor='k', label='Gaussian Means ($\mu$)', zorder=3)
    
    plt.title("Gaussian Mixture Model (GMM)\nAdapting to Elliptical Data with Probability Contours")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('clustering/gmm_cluster.jpg')
def plot_with_colors(X,y_true):
    # 1. Generate standard spherical blobs


    # 2. Stretch the blobs to make them elliptical
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_stretched = np.dot(X, transformation)

    # 3. Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_stretched)

    # 4. Visualize
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.get_cmap('tab10')

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(X_stretched[mask, 0], X_stretched[mask, 1],
                c=[cmap(label)], s=30, alpha=0.8, label=f'Cluster {label}')

    ax.set_title('GMM Clustering on Stretched Blobs')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.tight_layout()
    plt.savefig('gmm.jpg')
    plt.savefig('clustering/gmm_color.jpg')

np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
# Run the visualization
plot_gmm_ellipses(X,y_true)


plot_with_colors(X,y_true)