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
		* ```SUM((X_1-\overline{X}_1)*(X_2-\overline{X]_2)*(X_n-\overline{X}_n)) \mid \Var(X_1)*\Var(X_2)*\Var(X_n) 
				```

		* Pearson's coefficient values vary between 1 and -1:
			* 1 is highly correlated: the more of x1, the more of x2
			* -1 is highly correlated: the more of x1,the less of x2

	* For implementation refer to Correlation_FE.ipynb and Filter_method.ipynb
