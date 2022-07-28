# New Methods  of Verifying Neural Networks

## Introduction
Deep learning, a subfield of machine learning, imitates the brain's use of neural networks in order to mimic how humans think and learn. In deep learning, the structural building blocks of deep learning can be referred to as a Perceptron (aka a single layer neural network). A perceptron is comprised of 4 parts: An Input (an input layer/ input values), Weight and Bias, The Weighted Sum, and an Activication Function. First, all inputs are multiplied by their weights (internal parameters that impact how a particular input figures into the result). The weighted sum is then added to a bias value that is meant to increase the accuracy of a particular model. Then, the results are then applied to an activation function. An activation function is critical to perceptrons because it introduces non-linearities in a neural network. The result of the activation function is the perceptron's output. Moreover, a deep neural network is a neural network with multiple layers, which are referred to as hidden layers, between the input layer and the output layer.
## Adversarial Examples and Previous Methods of Verification of Neural Networks 
In machine learning, a technique named Train-Valid-Test is often used to evaluate the performance of machine learning models. To test a model, there will be a dataset for training, another dataset for validation, and another dataset for testing. The training set , which should be diverse in terms of inputs, is used so the model can be trained to learn distinct features and patterns of the set. The validation set is used to validate the model's performance while tuning the model's hyperparameters and configurations. The testing set is the set used to determine if the model is robust because it gives an unbiased evaluation of the performance in terms of accuracy and precision.  

Deep Neural Networks are vulnerable to adversarial perturbations/examples which are inputs formed by applying small perturbations to examples from the dataset that the cause the model to output an incorrect answer with high confidence. It should be noted that this phenomenon is often attributed to linear behavior in high-dimensional space. As a result, the need for certifying the robustness of neural networks has led to the the development of multiples tools designed for the verification of Neural Networks. The use of Satisfiability Modulo Theories (SMT) solvers to verify neural networks proved to be effective, yet could be limited in terms of scalability as it only worked with small networks with tens of nodes. As a result, the development of tools such as Reluplux and Marabou focus on brigding the problem of scalability by allowing the testing of larger networks with hundred or a few thousand of nodes. Marabou, a recent SMT-based verification tool, expands on the tool Reluplux by adding features such as extension of multiple interfaces for feeding queries into a solver, accommodation of piecewise-linear activation functions, and a complete simplex-based linear programming core.
## Netron: Neural Network Visualization Tools
Due to the black-box nature and the complexities of neural networks, it is hard for human to visualize and conceptualize the neural networks.Using the an [ONNX Model](https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/onnx/fc1.onnx) given in the Marabou Directory, one can see the graphical representation of an ONNX Model.  Using the visualization tools such as [Netron](https://github.com/lutzroeder/netron), one can visualize and conceptualize the deep learning networks.
![alt text](https://github.com/ameenam2/Neural-Network-Verification-Tool/blob/main/Screen%20Shot%202022-07-28%20at%208.26.06%20AM.png)
In the model above, the input to the neural network is named "Placeholder:0". The dimension are listed as "unk__10*2" which means the tool is unable to recognize the dimensions of the given input. The computational node are labeled with B and C which shows the weight and bias of the neural network respectively. The computational nodes are then put into a RELU function. The process is repeated 3 times and the output is named "y_out:0". 
## Open Neural Network Exchange (ONNX) for  Machine Learning Models
Open Neural Network Exchange(ONNX) is an intermediary machine learning framework that allows conversion between different machine learning frameworks. These machine learning frameworks consist of an interface that makes it simpler and faster for developers to create and deploy machine learning models. Prior to ONNX, conversion of machine learning models from one framework to another was extremely difficult.
## ONNX Key Design Principles
- Allow for Deep Neural Network as well as traditional Machine Learning Models.
- Adaptable enough to keep up with rapid advances.
- Compact and cross-platform representation for serialization.
- A standardized list of well-defined operators based on real-world usage.
## ONNX File Format
- Model
  - Version Info 
  - MetaData
  - Acyclic Computation DataFlow Graph
 - Graph
   - Inputs and Outputs
   - List of Computational Nodes
   - Graph Name
## Marabou 
Marabou, the SMT verification tool, operates by answering queries about a network's properties by translating the queries into constraint satisfication problems. Marabou can respond to the following types of verification queries: 
Reachability queries: if inputs is in a given range is the output guaranteed to be in some, typically safe, range.
Robustness queries: test whether there exist adversarial points around a given input point that change the output of the network.

### Verification of Neural Networks
When verifying a neural network, the query comprises of 2 parts: the neural network **N** and the property **P**. The Property **P** is in the form of P<sub>in</sub> ⇒ P<sub>out</sub>, where P<sub>in</sub> denotes a formula over N's inputs and P<sub>out</sub> denotes a formula over N's outputs. Typically, P<sub>in</sub> defines an input region I, and P states that P<sub>out</sub> holds for the output layer at each point in I. A verification tool will try to find a counter-example to this query: an input point I in I such that when applied to N, P<sub>out</sub> is false over the resulting outputs. P is true only if no such counter-example exists.

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
As mentioned previously, to make a verification query in Marabou, two parts are needed: a neural network and a property to verify. Depending on the format of the neural network, Marabou will work differently from the command line and the Python API.

### NNET Format 
To make a query for a DNN in the nnet format in Marabou, the properties can specified using inequalities over the input and output in a [.txt](https://github.com/NeuralNetworkVerification/Marabou/wiki/Marabou-Input-Formats) file placed in the resource folder:

To run the Marabou analysis from the command line on MAC OS,
```
cd path/to/marabou/repo/folder
./build/Marabou path/to/neural/network  path/to/property/file
```
```
./build/Marabou resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet resources/properties/acas_property_3.txt
```
Make sure to place both files in the resource folder located in the Marabou directory

## Maraboupy- The Python Interface of Marabou 
To make a query for a DNN in the onnx format in Marabou, the properties can be specified through the [Python API](https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/runMarabou.py#L80-L81) :

To run from the python script, one must export the Python and Jupyter paths using 
Run the following command: 
```
sudo nano ~/.bash_profile
```
Copy the following into the nano file: 
```
PYTHONPATH="/path/to/Marabou"
export PYTHONPATH
JUPYTER_PATH="/path/to/Marabou"
export JUPYTER_PATH
```
Then, run the following command:
```
source ~/.bash_profile
```
To properly install the Python Interface of Marabou on MAC OS, follow the following steps:
```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd path/to/marabou/repo/folder
mkdir build 
cd build 
cmake .. -DBUILD_PYTHON=ON
cmake --build .
```
To ensure that you correctly build Marabou using the correct path to Python,
```
which python
** copy the path/to/python
cmake .. -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=path/to/python
```
Also, make sure to export the PYTHON and JUPYTER PATHS as shown  above.
## Testing from the Marabou Python Interface

```
pip install pytest numpy ## Make sure pytest and numpy are both downloaded 
```
From the Marabou root directory, run the following command to install all other packages to 
```
pip install -r maraboupy/test_requirements.txt ##To download all the packages to make Marabou efficiently run
```
To test a file, place the python script in the Marabou/maraboupy/test folder and run the following command 
```
python -m pytest test               ## command to test the build 
python -m pytest test/test_nnet.py ##format: python -m pytest test/name_of_pythonfile.py

```
## Errors while Testing
When testing a python script using the command "python -m pytest test/test_nnet.py",
one may have errors as below shown: 
```
==================================== ERRORS ====================================
______________________ ERROR collecting test/test_nnet.py ______________________
ImportError while importing test module '/Users/ameenamohammed/Marabou/maraboupy/test/test_nnet.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
test/test_nnet.py:9: in <module>
    from .. import Marabou
Marabou.py:17: in <module>
    from maraboupy.MarabouCore import *
E   ImportError: No module named MarabouCore
!!!!!!!!!!!!!!!!!!! Interrupted: 1 errors during collection !!!!!!!!!!!!!!!!!!!!
=========================== 1 error in 0.04 seconds ============================
```
```
==================================== ERRORS ====================================
______________________ ERROR collecting test/test_nnet.py ______________________
ImportError while importing test module '/Users/ameenamohammed/Marabou/maraboupy/test/test_nnet.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
test/test_nnet.py:9: in 
from .. import Marabou
Marabou.py:17: in 
from maraboupy.MarabouCore import *
E ImportError: dlopen(/Users/ameenamohammed/Marabou/maraboupy/MarabouCore.cpython-39-darwin.so, 0x0002): tried: '/Users/ameenamohammed/Marabou/maraboupy/MarabouCore.cpython-39-darwin.so' (mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64'))
=========================== short test summary info ============================
ERROR test/test_nnet.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.07s ===============================
```
To resolve this issue, 


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
