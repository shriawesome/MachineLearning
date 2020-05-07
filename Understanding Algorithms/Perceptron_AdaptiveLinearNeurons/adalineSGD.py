import numpy as np

class AdalineSGD:
    """ ADAptive LInear NEuron classifier

        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
        shuffle : bool(default: True)
            Shuffles the training data every epoch if true to prevent cycles.
        random_state : int
            Random number generator seed for random weight initialization

        Attributes
        ------------
        w_ : 1d-array
            Weights after fitting.
        cost_ : list
            Sum-of-squares cost function value averaged over all training
            samples in each epoch.

    """

    def __init__(self,eta=0.01,n_iter=50,shuffle=True,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.shuffle=True
        self.random_state=1

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
        self._initialize_weights(X.shape[1])
        self.cost_=[]

        for _ in range(self.n_iter):
            if self.shuffle :
                X,y=self._shuffle(X,y)
            cost=[]

            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            self.cost_.append(sum(cost)/len(y))
        return self

    def _shuffle(self,X,y):
        """Shuffles training data to avoid Cycles."""
        r=self.rgen.permutation(len(y))
        return X[r],y[r]


    def _initialize_weights(self,m):
        """Initialise weights to small random numbers"""
        self.rgen=np.random.RandomState(self.random_state)
        self.w_=self.rgen.normal(loc=0.0,scale=0.01,size=m+1)
        self.w_initialized=True

    def _update_weights(self,xi,target):
        """Apply Adaline learning rule to update weights"""
        net_output=np.dot(xi,self.w_[1:])+self.w_[0]
        error=(target-net_output)
        self.w_[1:]+=self.eta*np.dot(xi,error)
        self.w_[0]+=self.eta*error
        cost=0.5*error**2
        return cost

    def predict(self,X):
        """Return the class label after the unit step"""
        net_output=np.dot(X,self.w_[1:])+self.w_[0]
        return np.where(net_output>=0.0,1,-1)
