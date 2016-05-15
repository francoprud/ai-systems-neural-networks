% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
%   {1}: matrix NxM where N is the amount of inputs and M is the size of each input
%   {2}: matrix NxK where N is the amount of outputs (equal to inputs) and K is the size of each output
% minimumError: minimum error that determines when to stop training
% learingRate: positive parameter for gradient descent to work
% layersAndSize: array that represents amount of layers and neurons in each layer, starting from the first hidden layer
function output = multilayer_perceptron(trainingSet, layersAndSize, minimumError, learingRate, activation_func, betha)
  totalInputs = rows(trainingSet{1});
  totalOutputs = rows(trainingSet{2});
  inputSize = columns(trainingSet{1});

  if (totalInputs != totalOutputs)
    printf('The training set is invalid.\n');
    return;
  end

  networkWeights = networkUtils.randomizeNetworkWeights([inputSize layersAndSize]);
  totalConnectionLayers = size(networkWeights)(2); % Total connections between layers

  V = forward_propagation(trainingSet{1}, totalConnectionLayers, networkWeights, activation_func, betha);
end
