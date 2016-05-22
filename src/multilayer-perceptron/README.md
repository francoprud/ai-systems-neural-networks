[README under construction...]

# Multilayer Perceptron

### Basic examples

```matlab
test_network.with_terrain('../../doc/data/terrain8.txt', 1, 3, [6 7 7 1], activation_functions.tanh, activation_functions.tanh_derived, 0.001, 0.02, 0.8, 0, 0.2, 0.2, 2)
test_network.with_function([6 5 4 1], @and, 4, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1, 0, 0.2, 0.2, 2)
test_network.with_function([3 2 1], @or, 4, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1, 0, 0.2, 0.2, 2)
test_network.with_function([3 1], @simetry, 3, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1, 0, 0.2, 0.2, 2)
test_network.with_function([3 4 1], @parity, 3, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1, 0, 0.2, 0.2, 2)
```
