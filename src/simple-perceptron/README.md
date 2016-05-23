# Simple Perceptron

#### Usage

```matlab
test_network(func, n, minimumError, learningRate, activation_func, activation_eval)
```

#### Arguments

- ***func*** [function]: function that want to be learned. It can be: *@and, @or*.
- ***n*** [integer]: the amount of columns of each input.
- ***minimumError*** [double]: minimum acceptable error at which the neural network will stop the learning.
- ***learningRate*** [double]: etha.
- ***activation_func*** [function]: activation function. Can be: *sigmoid.function, step.function or linear.function*.
- ***activation_eval*** [function]: the corresponding evaluation function for the activation function. If *sigmoid.function* then *sigmoid_eval*, if *step.function* then *step.eval*, and if *linear.function* then *linear.eval*.

### Basic examples

```matlab
test_network(@and, 3, 0.001, 0.5, sigmoid.function, sigmoid.eval)
test_network(@and, 3, 0.001, 0.5, linear.function, linear.eval)
test_network(@and, 3, 0, 0.5, step.function, step.eval)

test_network(@or, 3, 0.001, 0.5, sigmoid.function, sigmoid.eval)
test_network(@or, 3, 0.001, 0.5, linear.function, linear.eval)
test_network(@or, 3, 0, 0.5, step.function, step.eval)
```
