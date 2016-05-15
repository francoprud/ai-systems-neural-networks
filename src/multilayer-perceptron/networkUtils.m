function networkUtils = networkUtils()
  networkUtils.randomizeNetworkWeights = @randomizeNetworkWeights;
  networkUtils.forward_propagation = @forward_propagation;
end

function randomNetworkWeights = randomizeNetworkWeights(layersAndSize)
  for i = 1:(columns(layersAndSize) - 1)
    firstNeuronsAmount = layersAndSize(i) + 1;
    lastNeuronsAmount = layersAndSize(i + 1);

    randomNetworkWeights{i} = rand(firstNeuronsAmount, lastNeuronsAmount).-0.5; % Randomize between [-0.5, 0.5]
  end
end

function neuralValues = forward_propagation(inputs, totalConnectionLayers, networkWeights, activation_func, betha)
  totalInputs = rows(inputs);
  currentLayer = inputs;

  for i = 1:totalConnectionLayers
    layer = [ones(totalInputs, 1).*(-1) currentLayer];
    V{i} = activation_func(layer * networkWeights{i}, betha); % Hidden neurons
    currentLayer = V{i};
  end

  neuralValues = V;
end

