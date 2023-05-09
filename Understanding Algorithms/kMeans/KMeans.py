import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=5, max_iter=100, plot_steps=False):
        self.k = k
        self.max_iter = max_iter
        self.plot_steps = True
        self.centroids=[]
        
    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        # Randomly initialise centroids to random X points
        random_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = self.X[random_idx]
        # Iterate till max_iter or convergence
        for _ in range(self.max_iter):
            # Assign to clusters based on centroids
            clusters = self._assign_cluster(self.centroids)
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(clusters)
            # Plot intermediate steps
            if self.plot_steps:
                self._plot_step(clusters)
            # if converged
            if (np.all(self.centroids == old_centroids)):
                break
        
    def _assign_cluster(self, centroids):
        dist = np.zeros((self.n_samples, self.k))
        for k in range(self.k):
            dist[:,k] = np.linalg.norm(self.X - centroids[k,:], axis=1)
        return np.argmin(dist, axis=1)
    
    def _update_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for k in range(self.k):
            centroids[k,:]=np.mean(self.X[clusters==k,:], axis=0)
        return centroids
                
    def predict(self, x):
        # compute dist
        dist = np.zeros((x.shape[0], self.k))
        for k in range(self.k):
            dist[:,k] = np.linalg.norm(x - self.centroids[k,:], axis=1)
        return np.argmin(dist, axis=1)
    
    def _plot_step(self, clusters):
        for k in range(self.k):
            plt.scatter(self.X[clusters==k,0], self.X[clusters==k,1])
            plt.scatter(self.centroids[k,0], self.centroids[k,1], marker='x', color='black')
        plt.show()