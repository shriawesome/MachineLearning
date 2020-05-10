## Rosenblatt's threshold Perceptron Model  
- - -

  * Rosenblatt proposed an algorithm that would automatically learn the optimal weight coefficients that are then multiplied with the input features in order to make the decision of whether a neuron fires or not.
  * The decision function <img src="http://latex.codecogs.com/svg.latex?\phi(z)" border="0"/> takes a linear combination of certain input values x and a corresponding weight vector w, where z is the so-called net input :
      <br>  z= <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{1}x\textsubscript{1}" border="0"/> <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{2}x\textsubscript{2}" border="0"/> +... + <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{m}x\textsubscript{m}" border="0"/>

  * ![Vector Representation](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/imgs/vectors.png)

  * Now, if the net input of a particular sample is greater than a defined threshold, we predict class 1, and class -1 otherwise.
  * In the perceptron algorithm, the decision function is a variant of a unit step function.
  * ![Unit Step f'n](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/imgs/d_1.png)
  * For simplicity we can bring the threshold <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/> on the left hand side and define a weight 0 as -<img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/> and <img src="http://latex.codecogs.com/svg.latex?\x\textsubscript{0}" border="0"/> as 1, hence the equation can be written as :
  <br> z= <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{0}x\textsubscript{0}" border="0"/> <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{1}x\textsubscript{1}" border="0"/> +... + <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{m}x\textsubscript{m}" border="0"/>=<img src="http://latex.codecogs.com/svg.latex?w\textsupscript{T}x" border="0"/>
  * And :
  <br> ![Revised Decision F'n](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/imgs/d_1_1.png)
  * In machine learning, the term <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{0}" border="0"/>=-<img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/> is said to be a **Bias unit**.
