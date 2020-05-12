# Adaline refers to :- ADaptive LInear NEuron by Bernard Widrow
import numpy as np

class LogisticRegressionGD:
    """ LogisticRegression classifier

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
        cost_ : list
            value evaluated by logistic regression cost function value in each epoch.

    """

    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

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
        regn=np.random.RandomState(self.random_state)
        self.w_=regn.normal(loc=0.0,scale=0.01,size=1+X.shape[1])

        # Stores the cost values after each epoch
        self.cost_=[]

        for i in range(self.n_iter):
            # Unlike Perceptron we'll calculate the net_input for all the n_samples
            net_input=np.dot(X,self.w_[1:]) + self.w_[0]
            output=self.activation(net_input)
            errors=(y-output)
            self.w_[1:]+=self.eta*np.dot(X.T,errors)     # Transpose is needed so as to make matrix multiplication possible
            self.w_[0]+=self.eta*errors.sum()
            cost=(-y.dot(np.log(output)))-((1-y).dot(np.log(1-output)))
            self.cost_.append(cost)

        return self

    def activation(self,z):
        return 1./(1.+np.exp(-z))

    def predict(self,X):
        """Returns the class label after unit step"""
        net_input=np.dot(X,self.w_[1:]) + self.w_[0]
        return np.where(net_input>=0.0,1,0)
