function testOutput = test_network(func, N, activationFunc, activationFuncDerived, minimumError, learningRate, betha)
  dataSets = utils.get_random_subset(load('../../doc/data/terrain8.txt'), 1);
  trainingSet = dataSets{1};
  testingSet = dataSets{2};
  % trainingSet = utils.get_training_set(func, N);

  layersAndSize = [3 2 1];

  networkWeights = multilayer_perceptron_incremental(trainingSet, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha);
  for i = 1:rows(trainingSet{1})
    V = network_utils.forward_propagation([trainingSet{1}(i, :)], networkWeights, activationFunc, betha);
    trainingSet{1}(i, :)
    V{columns(layersAndSize)}
    % V{columns(layersAndSize)} >= 0.5
  end
end
