function testOutput = test_network(func, activationFunc, activationFuncDerived, minimumError, learningRate, betha)
  inputs = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
  outputs = func(inputs);
  trainingSet{1} = inputs;
  trainingSet{2} = outputs;

  layersAndSize = [2 1];

  networkWeights = multilayer_perceptron_batch(trainingSet, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha);
  for i = 1:rows(inputs)
    V = network_utils.forward_propagation([inputs(i, :)], networkWeights, activationFunc, betha);
    inputs(i, :)
    V{columns(layersAndSize)} >= 0.5
  end
end
