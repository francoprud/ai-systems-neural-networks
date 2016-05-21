% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
%   {1}: matrix NxM where N is the amount of inputs and M is the size of each input
%   {2}: matrix NxK where N is the amount of outputs (equal to inputs) and K is the size of each output
% minimumError: minimum error that determines when to stop training
% learingRate: positive parameter for gradient descent to work
% layersAndSize: array that represents amount of layers and neurons in each layer, starting from the first hidden layer
% activation_func:
% activation_func_derived:
% betha:
function output = multilayer_perceptron_incremental(trainingSet, layersAndSize, minimumError, learingRate, activation_func, activation_func_derived, betha)
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
  currentError = network_utils.calculate_error(trainingSet{2}, V{totalEdgesLayers});

  epoch = 0;
  currentMinimumError = currentError;

  while (currentError > minimumError)
    inputsOrder = randperm(totalInputs); % Randomize the order of inputs

    for i = 1:totalInputs
      currentIndex = inputsOrder(i);
      currentInput = trainingSet{1}(currentIndex, :);
      V = network_utils.forward_propagation(currentInput, networkWeights, activation_func, betha);

      outputLayerIndex = totalEdgesLayers;
      currentExpectedOutput = trainingSet{2}(currentIndex, :);
      currentOutput = V{outputLayerIndex};
      difference = (currentExpectedOutput - currentOutput);
      delta{outputLayerIndex} = activation_func_derived(currentOutput, betha).*difference;

      for j = (totalEdgesLayers):-1:2
        layers = networkWeights{j}(2 : end, :); % Remove -1 neuron to layers
        delta{j - 1} = activation_func_derived(V{j - 1}, betha).*(delta{j} * layers');
        networkWeights{j} = networkWeights{j} + learingRate * [-1 V{j - 1}]' * delta{j};
      end
      networkWeights{1} = networkWeights{1} + learingRate * [-1 trainingSet{1}(currentIndex, :)]' * delta{1};
    end

    V = network_utils.forward_propagation(trainingSet{1}, networkWeights, activation_func, betha);
    currentError = network_utils.calculate_error(trainingSet{2}, V{totalLayers -1});

    if (currentError < currentMinimumError)
      currentMinimumError = currentError;
    end

    epoch++;

    if (mod(epoch, 20) == 0)
      printf('epocas = %d; currentError = %g; currentMinimumError = %g\n', epoch, currentError, currentMinimumError);
      fflush(stdout);
    end
  end

  output = networkWeights;
end
