# Feature Extraction

1. FILTER METHODS - CORRELATION
	* 'Correlation' is a measure of the linear relationship of 2 or more variables
	* Through Correlation we can predict one variable from the other
		* Good variables are highly correlated with the target.
	* Correlated predictor variables provide redundant information
		* Variables should be highly correlated with the target but uncorrelated among themselves
	* `The central hypothesis is that good features sets contains features that are highly correlated with the class, yet uncorrelated with each other.`

	* CORRELATION AND ML
		* Correlated features do not necessarily affect model accuracy.
		* If two features are highly correlated, other would add a lil information over the previous: removing it helps reduce DIMENSIONALITY
		* Reduced High Dimensionality makes model more interpretable
		* Different Classifiers show different sensitivity to correlation.

	* PERSON'S CORRELATION COEFFICIENT
	 	* ```
       	      SUM((X1-X1mean)*(X2-X2mean)*(Xn-Xnmean)) / Var(X1)*Var(X2)*Var(Xn) 
       		``` 
		* Pearson's coefficient values vary between 1 and -1:
			* 1 is highly correlated: the more of x1, the more of x2
			* -1 is highly correlated: the more of x1,the less of x2

	* For implementation refer to Correlation_FE.ipynb and Filter_method.ipynb

	--- 
2. Statistical Methods with Filter Methods -
	Several methods that rely on Filter methods for Feature Selection :
		Statistical Methods
		|- Information Gain
		|- Fisher Score
		|- Univariate Tests
		|- Univariate roc-auc/rmse

		[Note: 
		1. None of the above methods take into consideration variable redundancy.
		2. All the dependency is checked in the light of target variables.]

	* Information Gain / Mutual Information
		* Mesures how much information the presence/absence of a feature contributes to making the correct prediction on Y.
		* MI measures the information that X and Y share: how much knowing one can reduce uncertainty about the other.
		* If X and Y are independent, then knowing X does not give any inforamation about Y and vice versa.
		* MI is the same as the ```entropy of X``` and as the ```entropy of Y```.
		* Popular scikit learn modules :-
			* ```sklearn.feature_selection - mutual_info_classify and mutual_info_regression```		

	--- 
