function networkUtils = network_utils()
  networkUtils.randomize_network_weights = @randomize_network_weights;
  networkUtils.forward_propagation = @forward_propagation;
  networkUtils.calculate_error = @calculate_error;
end

function randomNetworkWeights = randomize_network_weights(layersAndSize)
  for i = 1:(columns(layersAndSize) - 1)
    firstNeuronsAmount = layersAndSize(i) + 1;
    lastNeuronsAmount = layersAndSize(i + 1);

    randomNetworkWeights{i} = rand(firstNeuronsAmount, lastNeuronsAmount).-0.5; % Randomize between [-0.5, 0.5]
  end
end

function neuralValues = forward_propagation(inputs, networkWeights, activation_func, betha)
  totalConnectionLayers = size(networkWeights)(2); % Total connections between layers
  totalInputs = rows(inputs);
  currentLayer = inputs;

  for i = 1:totalConnectionLayers
    layer = [ones(totalInputs, 1).*(-1) currentLayer];
    V{i} = activation_func(layer * networkWeights{i}, betha); % Neurons biases including last layer
    currentLayer = V{i};
  end

  neuralValues = V;
end

function err = calculate_error(totalConnectionLayers, expectedOutputs, currentOutputs)
  differences = expectedOutputs - currentOutputs;
  err = sum(differences.**2) / (2 * totalConnectionLayers);
end