% func: function to test
% n: amount of bits
% minimumError: minimum error that determines when to stop training
% learingRate: positive parameter for gradient descent to work
% activation_func: activation function (logistic function/transference function)
% activation_eval: evaluate the activation function result
function output = test_network(func, n, minimumError, learningRate, activation_func, activation_eval)
  trainingSet = get_training_set(func, n);
  inputs = trainingSet{1};

  networkWeights = neural_network(trainingSet, minimumError, learningRate, activation_func);
  
  %randInput = inputs(randi(rows(inputs)), :)
  %output = evaluate_network(networkWeights, randInput, activation_func, activation_eval);

  output{1} = inputs(1, :);
  output{2} = evaluate_network(networkWeights, inputs(1, :), activation_func, activation_eval);
  for i = 2:rows(inputs)
    output{1} = [output{1}; inputs(i, :)];
    output{2}(i) = evaluate_network(networkWeights, inputs(i, :), activation_func, activation_eval);
  end
  output{2} = output{2}';
end
