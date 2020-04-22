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
	<br>	Statistical Methods
	<br>	|- Information Gain
	<br>	|- Fisher Score
	<br>	|- Univariate Tests
	<br>	|- Univariate roc-auc/rmse

			[Note:
			1. None of the above methods take into consideration variable redundancy.
			2. All the dependency is checked in the light of target variables.]

		* Information Gain / Mutual Information
			* Measures how much information the presence/absence of a feature contributes to making the correct prediction on Y.
			* MI measures the information that X and Y share: how much knowing one can reduce uncertainty about the other.
			* If X and Y are independent, then knowing X does not give any inforamation about Y and vice versa.
			* MI is the same as the ```entropy of X``` and as the ```entropy of Y```.
			* Popular scikit learn modules :-
				* ```sklearn.feature_selection - mutual_info_classify and mutual_info_regression```

		* Fisher Score - Chi-square implementation in sklearn
			* This score should be used to evaluate `Categorical Variables` in a `Classification` tasks.
			* Compute chi-squared stats between each non-negative feature and Class.
			* It compares observed distribution of the different classes of Y among different categories of the feature, against the expected distribution of the target classes, regardless of the feature categories.
			* Popular scikit learn modules :-
				* `sklearn.feature_selection -  chi2`

		* Univariate Feature Selection(ANOVA)
			* It works by selecting the features based on univariate statistical tests(ANOVA).
			* The methods based on F-test estimates the degree of linear dependency between the feature and the target.
			* These methods also assume that the variables follow Gaussian distribution.
			* Popular scikit learn modules :-
				* `sklearn.feature_selection - f_classif,f_regression`

		* Univariate ROC-AUC or MSE
			* First it works by building a ML model with the given feature, to predict the target.
			* Typically decision tree is used for predicting in classification problem.
			* Second, it makes predictions using the model and the mentioned feature.
			* Third, it ranks the features accoding to the ML metric(roc-aur or mse).
			* It then selects the highest ranked features
			* Popular scikit learn modules :-
				* `sklearn.metrics - roc_auc_score, mean_squared_error`

	---

2. WRAPPER METHODS FOR FEATURE SELECTION -
	* Can also be termed as Greedy Algorithm for feature selection.
	* Selects a feature by optimising performance of a specific ML model.
	* If follows a sequential process to select the features i.e. sequential FS algorithm adds/removes feature at the time based on classifier performance
		until a feature subset of the desired size 'k' is reached, or any other criteria is met.
	* 3 techniques broadly used in wrapper method:-
<br> |- Step Forward Feature selection
<br> |- Step Backward Feature selection
<br> |- Exhaustive Feature Selection

	* Step Forward Feature Selection method Procedure:
		1. Evaluate all subsets of 1 features.
		2. Choose the one that performs the best.
		3. Evaluate all the subsets of 2 features(the first selected and other).
		4. Evaluate algorithm performance.
		5. Repeat until criteria is met.

	* Step backward works in the same fashion but by removing the feature one by one.
	* Exhaustive FS :
		1. Makes all possible feature subsets from 1 to n(Total features).
		2. The one with the best performance is selected.

	* DISADVANTAGE :-
		* Feature space optimised for specific algorithm.
		* Extremely computationally expensive.
		* Often not feasible because of number of features.


  ---

3. EMBEDDED METHOD FOR FEATURE SELECTION -
	* Embedded method usually consists of Regularisation techniques and Tree importance for feature selection.
	* General procedure involves :-
		* Train a machine learning models.
		* Derive the feature importance.
		* Remove non-important features.

	1. Regularisation :-
		* There are basically 3 Regularisation techniques widely used :
	<br>|- Lasso (l1 Regularisation)
	<br>|- Ridge (l2 Regularisation)
	<br>|- Elastic net (l1/l2 Regularisation)
		* All the above techniques have an additional penalty term associated with the cost function along with the original measure of fit.
			* Lasso contains a penalty term : <img src="http://latex.codecogs.com/svg.latex?\lambda*|\theta|" border="0"/>
			* Ridge contains a penalty term : <img src="http://latex.codecogs.com/svg.latex?\lambda*(\theta)^2" border="0"/>

		1. L1(Lasso) Regularisation :
			* This can be used for feature selection from a given set of features for linear models.
			* It reduces the feature coefficient/weights(i.e. <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/>) to exactly 0.
			* It can be used to tackle the problem of `Bias-Variance` tradeoff.
				* If the value of <img src="http://latex.codecogs.com/svg.latex?\lambda" border="0"/> is high the the value of <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/>'s would be less or may be 0 hence may result in Bias problem with feature sparsity.
				* If the value of <img src="http://latex.codecogs.com/svg.latex?\lambda" border="0"/> is small then the value of <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/>'s would be large and hence may result in Variance problem with large number of features.
				* This inverse relationship can be attributed to The `Cost function` as we try to minimise the function.
			* Because of pt. 2 above it can be used for feature selection task.

		2. L2(Ridge) Regularisation :
			* Ride regularisation can't be used for feature selection as it doesn't reduce the value of <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/> to 0 but reduces to very small values.
			* Because of this we have large features of very less importance as well for the task of prediction.
			* It can be used to improve the performance of the model.

	2. Feature Selection based on coefficient values in Linear Models :-
		* Linear regression is a straight forward way of predicting the target based on different features.
		* Also it assumes that there is a linear relationship between the target(Y) and the inputs(X) i.e. Y=<img src="http://latex.codecogs.com/svg.latex?\theta\textsubscript{0}" border="0"/> +<img src="http://latex.codecogs.com/svg.latex?\theta\textsubscript{1}X\textsubscript{1}" border="0"/>+...+<img src="http://latex.codecogs.com/svg.latex?\theta\textsubscript{N}X\textsubscript{N}" border="0"/>
		* Hence the magnitude of the coefficients of X's directly influences the importance of that respective feature for prediction.
		* The magnitudes of the coefficients are directly affected by the scale of the corresponding feature.
		* Hence in case of linear models it is important to normalise the features into the same scale.
		* Assumptions that needs to be kept in mind for linear models w.r.t to X's are :
			* Linear relationship with the target i.e. Y.
			* It follows normal distribution.
			* X's should not be correlated among themselves.
			* Variance should be the same.

	3. SELECTING FEATURES BASED ON TREE IMPORTANCE :-
		* Trees like Decision Trees, Random Forests and even Gradient Boosted Trees can be used for the task of feature selection.
		* DTs are in general a sequence of decisions made on each node based on different feature at a time.
		* Feature is selected based on which particular feature reduces the impurity in the model.
		* This impurity is measured as :
			* For Classification :
		<br>|- Gini Index
		<br>|- Information gain
			* For Regression :
		<br>|- Variance
		* Random Forests are different than DTs in a way that it can in parallel create many different single trees, whereas in Gradient Boosted Trees many single trees are created based on improving performance from the previous trees.
		* In each of the above case different sets of features are used for creating trees.
		* A feature can be selected based on the importance of that feature in predicting the target.
		* Once the tree is trained it assigns a numerical value to each selected feature that is nothing but the importance of that feature.
		* Advantages :
			* Trees are very good at generalising(i.e. reducing Overfitting)
			* Trees work very good with the missing data and simply substituting a missing value with 0 or so it replaces it with an apt value.
			* Trees also work with the Categorical Variables.
			* Models created at the end are highly accurate.
			* Trees tend to reveal non-correlated relations between the features and the target.
		* Disadvantages :
			* Trees are biased towards highly Categorical Data.
			* Correlated features tend to show similar importance in predicting the target, which when used alone can high importance value in general.
			* Lesser amount of data when used for model development can often lead to Overfitting.
		* Procedure for feature Selection :
	<br> a. Build a random forest.
	<br> b. Determine feature importance.
	<br> c. Select the features with the highest importance.
