% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
%   {1}: matrix NxM where N is the amount of inputs and M is the size of each input
%   {2}: matrix NxK where N is the amount of outputs (equal to inputs) and K is the size of each output
% minimumError: minimum error that determines when to stop training
% learingRate: positive parameter for gradient descent to work
% layersAndSize: array that represents amount of layers and neurons in each layer, starting from the first hidden layer
% activation_func:
% activation_func_derived:
% betha:
function output = multilayer_perceptron_batch(trainingSet, layersAndSize, minimumError, learingRate, activation_func, activation_func_derived, betha)
  totalInputs = rows(trainingSet{1});
  totalOutputs = rows(trainingSet{2});
  inputSize = columns(trainingSet{1});
  totalLayers = columns(layersAndSize) + 1; % All layers including entrance and output layers
  totalEdgesLayers = columns(layersAndSize); % Total amount of edges "spaces"

  if (totalInputs != totalOutputs)
    printf('The training set is invalid.\n');
    return;
  end

  networkWeights = network_utils.randomize_network_weights([inputSize layersAndSize]);
  V = network_utils.forward_propagation(trainingSet{1}, networkWeights, activation_func, betha); % Contains the biases of hidden layers and output layers
  currentError = network_utils.calculate_error(totalEdgesLayers, trainingSet{2}, V{totalEdgesLayers});

  while (currentError > minimumError)
    delta{totalEdgesLayers} = activation_func_derived(V{totalEdgesLayers}, betha).*(trainingSet{2} - V{totalEdgesLayers});

    for j = totalEdgesLayers:-1:2
      layers = networkWeights{j}(2 : end, :); % Remove -1 neuron to layers
      delta{j - 1} = activation_func_derived(V{j - 1}, betha).*(delta{j} * layers');
      networkWeights{j} = networkWeights{j} + learingRate * [ones(totalInputs, 1).*(-1) V{j - 1}]' * delta{j};
    end
    networkWeights{1} = networkWeights{1} + learingRate * [ones(totalInputs, 1).*(-1) trainingSet{1}]' * delta{1};

    V = network_utils.forward_propagation(trainingSet{1}, networkWeights, activation_func, betha);
    currentError = network_utils.calculate_error(totalEdgesLayers, trainingSet{2}, V{totalEdgesLayers});
  end

  output = networkWeights;
end