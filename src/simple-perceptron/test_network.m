% func: function to test
% minimumError: minimum error that determines when to stop training
% learingRate: positive parameter for gradient descent to work
% activation_func: activation function (logistic function/transference function)
% activation_eval: evaluate the activation function result
function output = test_network(func, minimumError, learningRate, activation_func, activation_eval)
  inputs = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
  outputs = func(inputs);
  trainingSet{1} = inputs;
  trainingSet{2} = outputs;

  networkWeights = neural_network(trainingSet, minimumError, learningRate, activation_func);
  randInput = inputs(randi(rows(inputs)), :)
  output = evaluate_network(networkWeights, randInput, activation_func, activation_eval);
end
