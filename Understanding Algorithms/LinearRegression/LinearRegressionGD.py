import numpy as np

class LinearRegressionGD:
    def __init__(self, eta=0.001, n_iter=200):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # normal distribution random values for weights, adding 1 for the bias
        self.w_ = np.random.randn((1+X.shape[1]))
        self.cost_ = []
        for i in range(self.n_iter):
            # make predictions using weights
            z = self.net_input(X)
            # calculate the errors
            errors = (y-z)
            # compute the cost i.e. SSE(Sum of Squared Errors)
            cost = (errors**2).sum()/2
            # Update the weights using Gradient Descent
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return (np.dot(X, self.w_[1:]) + self.w_[0])
    
    def predict(self, X):
        return self.net_input(X)