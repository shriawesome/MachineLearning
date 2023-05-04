import numpy as np 

class SVM:
    def __init__(self, eta=0.001, lambda_pm = 0.01, n_iter=200) -> None:
        self.eta = 0.001
        self.lambda_pm = lambda_pm
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features  = X.shape
        y = np.where(y<=0,-1,1)

        # Initialise weights and biases to 0
        self.w = np.zeros((n_features))
        self.b = 0

        # Update the weigths for n_iter
        for _ in range(self.n_iter):
            z = np.dot(X, self.w) + self.b
            # checking where y.z >= 1
            cond_mask = y*z < 1
            
            # Updating the weights, using vectorisation
            # when y*z >= 1, no update to bias
            pos_cond = n_samples - cond_mask.sum()
            self.w -= self.eta*(2*self.lambda_pm*self.w*(pos_cond))

            # when y*z < 1
            self.w -= self.eta*(2*self.lambda_pm*self.w*cond_mask.sum() - np.dot(X[cond_mask].T, y[cond_mask]))
            self.b -= self.eta*y[cond_mask].sum()

    def predict(self, X):
        z = np.dot(X,self.w) - self.b
        return np.sign(z)
