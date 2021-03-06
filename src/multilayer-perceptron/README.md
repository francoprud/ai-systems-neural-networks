# Multilayer Perceptron

Change directory from the root of the project to ./src/multilayer-perceptron.

***To speed up the using of the network you can change the function params from the `run.m` file and then execute in octave `run`.***

The Neural Newtork can be executed with two methods:

### `test_network.with_function`

Given a function like *and, or, xor, parity and simetry* returns all the inputs evaluated by the neural network after learning all the weights of the connections.

#### Usage

```matlab
test_network.with_function(func, N, algorithm, improvement, layersAndSize, activationFunc, activationFuncDerived, minimumError, learningRate, betha, alpha, adaptativeA, adaptativeB, kEpochs)
```

#### Arguments

- ***func*** [function]: function that want to be learned. It can be: *@and, @or, @xor, @parity, @simetry*.
- ***N*** [integer]: the amount of columns of each input.
- ***layersAndSize*** [array]: array that represents the amount of layers and neurons in each layer, starting from the first hidden layer.
- ***activationFunc*** [function]: activation function. Can be: *activation_functions.tanh*, *activation_functions.exp*.
- ***activationFuncDerived*** [function]: derived function of the activation function. If *activation_functions.tanh* then it must be *activation_functions.tanh_derived*. If *activation_functions.exp* then it must be *activation_functions.exp_derived*.
- ***minimumError*** [double]: minimum acceptable error at which the neural network will stop the learning.
- ***learningRate*** [double]: etha.
- ***betha*** [double]: parameter for the activation function.
- ***alpha*** [double]: parameter for the momentum improvement.
- ***adaptativeA*** [double]: parameter for the adaptative etha improvement. It is the increment of etha.
- ***adaptativeB*** [double]: parameter for the adaptative etha improvement. It is the percentage of the decrement of etha.
- ***kEpochs*** [integer]: parameter for the adaptative etha improvement. It is the amount of epochs that adaptative etha imporvements needs.

#### Basic examples

```matlab
test_network.with_function(@and, 3, [3 1], activation_functions.tanh, activation_functions.tanh_derived, 0.01, 0.5, 0.1, 0, 0.1, 0.1, 2)

test_network.with_function(@simetry, 3, [3 1], activation_functions.tanh, activation_functions.tanh_derived, 0.01, 0.1, 0.55, 0, 0.1, 0.1, 2)

test_network.with_function(@parity, 3, [6 1], activation_functions.tanh, activation_functions.tanh_derived, 0.01, 0.1, 0.55, 0.4, 0.1, 0.1, 2)
```

### `test_network.with_terrain`

Given a sample of values of a function (in this case to simulate a terrain), returns all the weights of the connections of the neural network.

#### Usage

```matlab
test_network.with_terrain(filePath, trainingPercentage, layersAndSize, activationFuncId, minimumError, learningRate, betha, alpha, adaptativeA, adaptativeB, kEpochs, plot)
```

#### Arguments

- ***filePath*** [string]: the path of the file to load the sample values.
- ***trainingPercentage*** [double]: the percentage of those same values to be consider as the training inputs, the rest will be consider as the testing input. Value from 0 to 1.
- ***layersAndSize*** [array]: array that represents the amount of layers and neurons in each layer, starting from the first hidden layer.
- ***activationFuncId*** [integer]: activation function. Can be: `1`: *activation_functions.tanh*, or `2`:  *activation_functions.exp*.
- ***minimumError*** [double]: minimum acceptable error at which the neural network will stop the learning.
- ***learningRate*** [double]: etha.
- ***betha*** [double]: parameter for the activation function.
- ***alpha*** [double]: parameter for the momentum improvement.
- ***adaptativeA*** [double]: parameter for the adaptative etha improvement. It is the increment of etha.
- ***adaptativeB*** [double]: parameter for the adaptative etha improvement. It is the percentage of the decrement of etha.
- ***kEpochs*** [integer]: parameter for the adaptative etha improvement. It is the amount of epochs that adaptative etha imporvements needs.
- ***plot*** [boolean]: `true` plots the graphics, `false` don't.

#### Basic examples

```matlab
test_network.with_terrain('../../doc/data/terrain8.txt', 1, [6 5 4 1], 1, 0.00001, 0.05, 0.75, 0, 0, 0, 0, true)

test_network.with_terrain('../../doc/data/terrain8.txt', 1, [10 18 7 1], 1, 0.0001, 0.05, 0.75, 0, 0, 0, 0, true)

test_network.with_terrain('../../doc/data/terrain8.txt', 1, [20 15 10 1], 1, 0.0001, 0.05, 0.75, 0.5, 0, 0, 0, false)
```
