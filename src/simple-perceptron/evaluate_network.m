% network: weights from training
% input: array to test
% func: activation function (logistic function/transference function)
function output = evaluate_network(network, input, func)
  output = func([-1 input] * network);
end
