# New Methods  of Verifying Neural Networks

## Introduction
Deep learning, a subfield of machine learning, imitates the brain's use of neural networks in order to mimic how humans think and learn. In deep learning, the structural building blocks of deep learning can be referred to as a Perceptron (aka a single layer neural network). A perceptron is comprised of 4 parts: An Input (an input layer/ input values), Weight and Bias, The Weighted Sum, and an Activication Function. First, all inputs are multiplied by their weights (internal parameters that impact how a particular input figures into the result). The weighted sum is then added to a bias value that is meant to increase the accuracy of a particular model. Then, the results are then applied to an activation function. An activation function is critical to perceptrons because it introduces non-linearities in a neural network. The result of the activation function is the perceptron's output. Moreover, a deep neural network is a neural network with multiple layers, which are referred to as hidden layers, between the input layer and the output layer.
## Adversarial Examples and Previous Methods of Verification of Neural Networks 
In machine learning, a technique named Train-Valid-Test is often used to evaluate the performance of machine learning models. To test a model, there will be a dataset for training, another dataset for validation, and another dataset for testing. The training set , which should be diverse in terms of inputs, is used so the model can be trained to learn distinct features and patterns of the set. The validation set is used to validate the model's performance while tuning the model's hyperparameters and configurations. The testing set is the set used to determine if the model is robust because it gives an unbiased evaluation of the performance in terms of accuracy and precision.  


Deep Neural Networks are vulnerable to adversarial perturbations/examples which are inputs formed by applying small perturbations to examples from the dataset that the cause the model to output an incorrect answer with high confidence. It should be noted that this phenomenon is often attributed to linear behavior in high-dimensional space. As a result, the need for certifying the robustness of neural networks has led to the the development of multiples tools designed for the verification of Neural Networks. The use of Satisfiability Modulo Theories (SMT) solvers to verify neural networks proved to be effective, yet could be limited in terms of scalability as it only worked with small networks with tens of nodes. As a result, the development of tools such as Reluplux and Marabou focus on brigding the problem of scalability by allowing the testing of larger networks with hundred or a few thousand of nodes. Marabou, a recent SMT-based verification tool, expands on the tool Reluplux by adding features such as extension of multiple interfaces for feeding queries into a solver, accommodation of piecewise-linear activation functions, and a complete simplex-based linear programming core.
## Open Neural Network Exchange (ONNX) for  Machine Learning Models
Open Neural Network Exchange(ONNX) is an intermediary machine learning framework that allows conversion between different machine learning frameworks. These machine learning frameworks consist of an interface that makes it simpler and faster for developers to create and deploy machine learning models. Prior to ONNX, conversion of machine learning models from one framework to another was extremely difficult.
### Onnx File Format
- Model
  - Version Info 
  - MetaData
  - Acyclic Computation DataFlow Graph
 - Graph
   - Inputs and Outputs
   - List of Computational Nodes
   - Graph Name

## Marabou 
Marabou, the SMT verification verification, operates by answering queries about a network's properties by translating the queries into constraint satsification problems.


### Verification of Neural Networks
When verifying a neural network, the problem comprises of 2 parts: the neural network **N** and the property **P**. The Property **P** is in the form of P<sub>in</sub> ⇒ P<sub>out</sub> ,where P<sub>in</sub> denotes a formula over N's inputs and P<sub>out</sub> denotes a formula over N's outputs. Typically, P<sub>in</sub> defines an input region I, and P states that P<sub>out</sub> holds for the output layer at each point in I. A verification tool will try to find a counter-example to this query: an input point I in I such that when applied to N, P<sub>out</sub> is false over the resulting outputs.P is true only if no such counter-example exists.

For more information, check out the following links
- [Marabou Github](https://github.com/NeuralNetworkVerification/Marabou) 
- [Marabou Paper](https://aisafety.stanford.edu/marabou/MarabouCAV2019.pdf) 
- [Marabou Documentation](https://neuralnetworkverification.github.io/Marabou/API/0_Marabou.html) 
## Installation Instructions for MAC OS
To build Marabou on MAC OS Terminal, Download [CMAKE](https://cmake.org/download/) 3.12 or later. Also, Download [Homebrew](https://phoenixnap.com/kb/install-homebrew-on-mac).

Then, use the following commands
```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd path/to/marabou/repo/folder
mkdir build
cd build
brew install boost
cmake ..
cmake --build .
```
It is to be noted that a difference in CPU architecture or recent MAC OS updates may impact the building process. It may generate errors as shown below ![alt text](https://user-images.githubusercontent.com/85082930/172899138-4d8c3517-6aa8-4962-9743-6eaabe861f45.png)

This issue can be resolved by following the the directions detailed [here](https://github.com/NeuralNetworkVerification/Marabou/issues/570).
## Command Interface of Marabou
As mentioned previously, to make a verfification query in Marabou, two parts are needed: a neural network and a property to verify. Depending on the format of the neural network, Marabou will work differently

### NNET Format 




### ONNX Format 




## References
- https://neuralnetworkverification.github.io/Marabou/API/0_Marabou.html
- https://arxiv.org/pdf/2004.08440.pdf
- https://aisafety.stanford.edu/marabou/MarabouCAV2019.pdf
- https://arxiv.org/pdf/1412.6572.pdf
- https://www.v7labs.com/blog/train-validation-test-set
- https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
- https://www.microsoft.com/en-us/research/video/onnx-and-onnx-runtime/
- https://static.linaro.org/connect/san19/presentations/san19-211.pdf
- https://github.com/NeuralNetworkVerification/Marabou.
