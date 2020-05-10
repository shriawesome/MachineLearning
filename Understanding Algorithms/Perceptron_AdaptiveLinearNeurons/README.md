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
  <br> z= <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{0}x\textsubscript{0}" border="0"/> <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{1}x\textsubscript{1}" border="0"/> +... + <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{m}x\textsubscript{m}" border="0"/>=<img src="http://latex.codecogs.com/svg.latex?w\textsuperscript{T}x" border="0"/>
  * And :
  <br> ![Revised Decision F'n](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/imgs/d_1_1.png)
  * In machine learning, the term <img src="http://latex.codecogs.com/svg.latex?w\textsubscript{0}" border="0"/>=-<img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/> is said to be a **Bias unit**.
  * The following diagram illustrates how the net input of the perceptron can be used for binary classification(1 or -1) with the linear decision function that can separate two linearly separable classes.
  * ![Explain](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/imgs/exp_1.png)
  * The Perceptron learning rule :
    - Initialise all the weights initially by either 0 or some small numbers.
    - Based on each observation <img src="http://latex.codecogs.com/svg.latex?\x\textsuperscript{i}" border="0"/> from the sample:
      * Predict the output using the **Unit Step** function.
      * Update all the weights accordingly.
  * Weights are updating using the following criteria
    <br><img src="http://latex.codecogs.com/svg.latex?\w\textsubscript{j}" border="0"/>  = img src="http://latex.codecogs.com/svg.latex?\w\textsubscript{j}" border="0"/>  + <img src="http://latex.codecogs.com/svg.latex?\deltaw\textsubscript{j}" border="0"/>
    <br> where <img src="http://latex.codecogs.com/svg.latex?\deltaw\textsubscript{j}" border="0"/> can be evaluated as :
    <br> <img src="http://latex.codecogs.com/svg.latex?\deltaw\textsubscript{j}" border="0"/>=<img src="http://latex.codecogs.com/svg.latex?\eta(y\textsuperscript{i}-y\textsubscript{pred}\textsuperscript{i})x\textsubscript{j}\textsuperscript{(i)}" border="0"/>
    <br> Where img src="http://latex.codecogs.com/svg.latex?\eta" border="0"/> is the learning rate(usually a constant between 0.0 and 1.0), img src="http://latex.codecogs.com/svg.latex?\y\textsuperscript{i}" border="0"/> is the **true class** label of the ith training sample.
  * The convergence of the ith training sample is only guaranteed if the two classes are linearly separable and the learning rate is comparatively small.
  * If the classes are not linearly separable we can define number of **Epochs** over the training samples, otherwise the perceptron will never stop updating the weights.
  * ![Explanation](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/imgs/exp_2.png)
  * For implementation refer to :
    * [Ipynb Code](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/Percepton_adalineGD_adalineSGD_implement.ipynb)
    * [Perceptron Python Code](https://github.com/shriawesome/MachineLearning/blob/master/Understanding%20Algorithms/Perceptron_AdaptiveLinearNeurons/basic_Perceptron.py)

- - -
### SOURCE :
  * [Python Machine Learning by Sebastian Raschka & Vahid Mirjalili]

### Contribution :
  * [Shrikant Kendre](https://www.linkedin.com/in/shrikant-kendre-2941a6143/)
- - -
