[README under construction...]

# Multilayer Perceptron

### Basic examples

```matlab
test_network.with_terrain([6 5 4 1], activation_functions.tanh, activation_functions.tanh_derived, 0.0005, 0.4, 0.5)
test_network.with_function([6 5 4 1], @and, 4, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
test_network.with_function([3 2 1], @or, 4, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
test_network.with_function([3 1], @simetry, 3, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
test_network.with_function([3 4 1], @parity, 3, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
```
