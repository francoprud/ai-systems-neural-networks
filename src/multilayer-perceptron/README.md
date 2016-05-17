[README under construction...]

# Multilayer Perceptron

### Basic examples

```matlab
test_network(@and, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
test_network(@or, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
test_network(@simetry, 3, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
test_network(@parity, 3, activation_functions.exp, activation_functions.exp_derived, 0.001, 0.5, 1)
```
