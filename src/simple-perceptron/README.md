[README under construction...]

# Simple Perceptron

### Basic examples

```matlab
test_network(@and, 0.001, 0.5, sigmoid.function, sigmoid.eval)
test_network(@and, 0.001, 0.5, linear.function, linear.eval)
test_network(@and, 0, 0.5, step.function, step.eval)

test_network(@or, 0.001, 0.5, sigmoid.function, sigmoid.eval)
test_network(@or, 0.001, 0.5, linear.function, linear.eval)
test_network(@or, 0, 0.5, step.function, step.eval)
```
linear function is not learning well (or taking a long time to do it). Need check.
