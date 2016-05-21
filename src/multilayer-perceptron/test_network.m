function testNetwork = test_network()
  testNetwork.with_function = @with_function;
  testNetwork.with_terrain = @with_terrain;
end

function withFunctionOutput = with_function(layersAndSize, func, N, activationFunc, activationFuncDerived, minimumError, learningRate, betha)
  trainingSet = utils.get_training_set(func, N);

  networkWeights = multilayer_perceptron_incremental(trainingSet, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha);
  for i = 1:rows(trainingSet{1})
    V = network_utils.forward_propagation([trainingSet{1}(i, :)], networkWeights, activationFunc, betha);
    trainingSet{1}(i, :)
    V{columns(layersAndSize)} >= 0.5
  end
end

function withTerrainOutput = with_terrain(layersAndSize, activationFunc, activationFuncDerived, minimumError, learningRate, betha)
  dataSets = utils.get_random_subset(load('../../doc/data/terrain8.txt'), 1);
  trainingSet = dataSets{1};
  testingSet = dataSets{2};

  networkWeights = multilayer_perceptron_incremental(trainingSet, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha);

  fid = fopen('../../doc/data/result.txt', 'w+');
  fprintf(fid, '%d\n\n', rows(trainingSet{1}));
  for i = 1:rows(trainingSet{1})
    V = network_utils.forward_propagation([trainingSet{1}(i, :)], networkWeights, activationFunc, betha);
    aux = trainingSet{1}(i, :);
    result = [trainingSet{1}(i, :) V{columns(layersAndSize)}];

    fprintf(fid, '%g ', result);
    fprintf(fid, '\n');
  end
  fclose(fid);
end
