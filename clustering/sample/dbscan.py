import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from pprint import pprint
def plot_dbscan_with_epsilon(X, labels, eps, points_to_highlight=None):
    """
    Plots DBSCAN clusters and draws the epsilon search radius around specific points.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique labels (trails + noise)
    unique_labels = set(labels)
    
    # Assign a color to each cluster, use black for noise (-1)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
            marker = 'x'        # X marker for noise
            label_name = 'Noise'
        else:
            marker = 'o'
            label_name = f'Trail {k}'

        # Find points belonging to this cluster
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        # Plot the points
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                   edgecolor='k', s=50, label=label_name)

    # If no specific points are provided, pick 5 random points to draw shadows around
    if points_to_highlight is None:
        np.random.seed(42)
        points_to_highlight = np.random.choice(range(len(X)), size=5, replace=False)

    # Draw the epsilon radius 'shadows' around the selected points
    for idx in points_to_highlight:
        point = X[idx]
        
        # Draw the center point a bit larger
        ax.scatter(point[0], point[1], c='red', s=100, zorder=5)
        
        # Draw the epsilon radius shadow
        circle = Circle((point[0], point[1]), eps, color='red', alpha=0.2, zorder=1)
        ax.add_patch(circle)
        
    ax.set_title(f"DBSCAN Clusters with $\epsilon={eps}$ Search Radius Shadows")
    ax.set_xlabel("Longitude (Mock)")
    ax.set_ylabel("Latitude (Mock)")
    ax.legend()
    
    # Equal aspect ratio ensures circles look like circles, not ovals
    ax.set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('clustering/dbscan_cluster.jpg')

def plot_dbscan_color(X_gps,labels,eps):
    od = {}
    for idx, label in enumerate(labels):
        key = str(int(label))          # convert ONCE, use consistently
        points = od.get(key, [])       # now the lookup key matches the stored key
        points.append(X_gps[idx])
        od[key] = points

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.get_cmap('tab10')

    for i, (label, points) in enumerate(od.items()):
        points_arr = np.array(points)
        if label == '-1':
            ax.scatter(points_arr[:, 0], points_arr[:, 1],
                    c='gray', s=20, alpha=0.4, label='Noise')
        else:
            ax.scatter(points_arr[:, 0], points_arr[:, 1],
                    c=[cmap(int(label))], s=30, alpha=0.8, label=f'Cluster {label}')

    ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples=10)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.tight_layout()
    plt.savefig('clustering/dbscan_color.jpg')



# --- Example Usage ---
# 1. Generate Data
X_gps, _ = make_moons(n_samples=300, noise=0.5, random_state=0)

eps_value = 0.2
dbscan = DBSCAN(eps=eps_value, min_samples=10)
labels = dbscan.fit_predict(X_gps)

# 3. Plot (It will highlight 5 random points and draw their eps radius)
plot_dbscan_with_epsilon(X_gps, labels, eps=eps_value)

plot_dbscan_color(X_gps, labels, eps=eps_value)