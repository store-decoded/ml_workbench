import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
# Mean Shift
def plot_mean_shift():
    # 1. Generate festival GPS pings
    # We will use an uneven number of points per cluster to simulate different crowd sizes
    np.random.seed(42)
    X_pings, _ = make_blobs(n_samples=[200, 100, 50], centers=[[2, 2], [6, 6], [8, 2]], 
                            cluster_std=[0.8, 0.6, 0.4], random_state=42)
    
    # 2. Estimate the Bandwidth (the "flashlight radius")
    # quantile=0.2 means we look at the distances of the nearest 20% of points to guess a good radius
    bw = estimate_bandwidth(X_pings, quantile=0.2, n_samples=100)
    print(f"Estimated Bandwidth: {bw:.2f}")
    
    # 3. Fit Mean Shift
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(X_pings)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    # 4. Setup the Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter the crowd colored by their final destination peak
    ax.scatter(X_pings[:, 0], X_pings[:, 1], c=labels, cmap='Set1', 
               s=20, alpha=0.6, edgecolor='none', zorder=2)
    
    # 5. Plot the Peaks (Cell Tower locations) and their Bandwidth attraction zones
    for i, center in enumerate(cluster_centers):
        # The peak
        ax.scatter(center[0], center[1], c='black', marker='*', s=300, 
                   edgecolor='white', zorder=4, label='Peak (Tower)' if i == 0 else "")
        
        # The Bandwidth circle (the final area of attraction)
        circle = plt.Circle((center[0], center[1]), bw, color='black', 
                            fill=False, linestyle='--', linewidth=2, alpha=0.5, zorder=3)
        ax.add_patch(circle)
    
    # Force equal aspect ratio so the bandwidth circles are perfectly round
    ax.set_aspect('equal')
    
    plt.title(f"Mean Shift Clustering\nAutomatically found {len(cluster_centers)} peaks with Bandwidth R = {bw:.2f}")
    plt.xlabel("GPS Longitude")
    plt.ylabel("GPS Latitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('clustering/ms.jpg')

# Run the visualization
plot_mean_shift()
