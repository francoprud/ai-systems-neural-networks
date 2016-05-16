function testOutput = test_network(func, activationFunc, activationFuncDerived, minimumError, learningRate, betha)
  inputs = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
  outputs = func(inputs);
  trainingSet{1} = inputs;
  trainingSet{2} = outputs;

  layersAndSize = [2 1];

  networkWeights = multilayer_perceptron_incremental(trainingSet, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha);
end
