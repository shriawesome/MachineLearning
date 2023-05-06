import numpy as np
import heapq

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X ,y):
        '''
        fit method in kNN just stores and remembers the training data to be used during the prediction
        '''
        self.x_train=X
        self.y_train=y
    
    def predict(self, x):
        # Take a datapoint and find the distance of all the points from x
        dists = [np.linalg.norm(x[row]-self.x_train, axis=1) for row in range(x.shape[0])]
        # Sort and Select top k values
        preds = []
        for row in range(len(dists)):
            # Below is a O(n) operation
            minHeap = dists[row].tolist()
            heapq.heapify(minHeap)
            count = 0
            while count!=self.k:
                k_min = heapq.heappop(minHeap)
                count += 1
            k_mask = dists[row]<=k_min
            # Taking the majority vote
            labels, counts = np.unique(self.y_train[k_mask], return_counts=True)
            idx_majority = np.argmax(counts)
            preds.append(labels[idx_majority])
            
        return np.array(preds)
            
            