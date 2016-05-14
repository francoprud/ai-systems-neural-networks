% network: weights from training
% input: array to test
% activation_func: activation function (logistic function/transference function)
% activation_eval: evaluate the activation function result
function output = evaluate_network(network, input, activation_func, activation_eval)
  output = activation_eval(activation_func([-1 input] * network)); % Returns 0 or 1
end
