# Neural-Network-Verification-Tool---Marabou

## Introduction
Deep learning, a subfield of machine learning, imitates the  brain's use of neural network in order to mimc how humans think and learn. In deep learning, the structural building block of deep learning can be referred to as a Perceptron(aka a single layer neural network). A perceptron is comprised of 4 parts: An Input (A input layer/ Input Values), Weight and Bias, The Weighted Sum, and an Activication Function. First, all inputs are multiplied by their weights (internal parameters that impact how a particular input figures into the result). The weighted sum is then added to a bias value that is meant to increase the accurarcy of a particular model.T Then, the results is then applied to a activation function. An activation is critical to the perceptron because it introduces non-linearities in a neural network. The result of the activation function is the perceptron's output. Moreover, a deep neural network is a neural network with multiple layers, which are refered to as hidden layers, between the input layer and the output layer. 
## Adversial Examples and Previous Methods of Verification of Neural Networks 
In machine learning, A techinque named Train-Valid-Test is often used to evaluate the performance of machine learning models. To test a model, there will be a dataset for training, another dataset for validation, and another datset for testing. The training set , which should be diverse in terms of inputs, is used so the model can be trained to learn distinct features and patterns of the set. The validation set is used to validate whether the model's perfomancw while tuning the model's hyperparameters and configurations. The Testing set is the set used to determine if the model is robust because it gives unbiased evaluation of the performance in terms of accuracy and precision.  

Deep Neural Networks are vulnerable to **adversarial perturbations/examples** which are inputs formed by applying small pertubations to examples from the dataset that the cause the model to output an incorrect answer with high confidence. It is be noted that this phenomenon is often attributed linear behavior in high-dimensional space. As a result, the need for certifying the robustness of neural networks has led to the the development of multiples tools designed for the verification of Neural Networks. The use of Satisfiability Modulo Theories (SMT) solvers to verify neural networks proved to effective yet can be limited in terms of scalability as it only worked with small networks with tens of nodes. As a result, the development of tools such as Reluplux and Marabou focus on brigding the problem of scalilbity by allowing the testing of larger networks with hundred or a few thousand of nodes. Marabou, a recent SMT-based verification tools, expands on the tool Reluplux by adding features such as extension of multiple interfaces for feeding queries into a solver, accomodation of piecewise-linear activation functions, and a complete simplex-based linear programming core. 
## Marabou 
Marabou, the SMT verification verification, operates by answring queries about a network's properties by translating the queries into constraint satsification problems. For more information, check out the following links
[Marabou Github](https://github.com/NeuralNetworkVerification/Marabou) 
[Marabou Paper](https://aisafety.stanford.edu/marabou/MarabouCAV2019.pdf) 
[Marabou Documentation](https://neuralnetworkverification.github.io/Marabou/API/0_Marabou.html) 
## Installation Instructions for MAC OS
To build Marabou on MAC OS Terminal, Download [CMAKE](https://cmake.org/download/) 3.12 or later.
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
## Structure of Marabou 

## Command Interface of Marabou

## References

