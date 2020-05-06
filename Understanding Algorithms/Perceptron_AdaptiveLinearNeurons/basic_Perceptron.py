import numpy as np

class Perceptron:
    """Perceptron Classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X,y):
        """ Fit training data.

        Parameters
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of Training samples
            and n_features is the number of features.
        y : {array-like}, shape =[n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        regn = np.random.RandomState(self.random_state)
        self.w_ = regn.normal(loc=0.0,scale=0.01,size=1+X.shape[1])  # gets random numbers from normal distribution with mean as 'loc' and std as 'scale'.

        self.error_=[]

        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target - self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update != 0.0)           # Keeps the count of how many updates had been made
            self.error_.append(errors)
        return self



    def predict(self,X):
        """Returns the class label after unit step"""
        net_input=np.dot(X,self.w_[1:])+self.w_[0]
        return np.where(net_input>=0.0,1,-1)    # works as if else statement.
