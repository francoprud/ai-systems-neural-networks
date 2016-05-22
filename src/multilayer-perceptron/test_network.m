function testNetwork = test_network()
  testNetwork.with_function = @with_function;
  testNetwork.with_terrain = @with_terrain;

  testNetwork.configurations = {
    {
      @multilayer_perceptron_incremental,
      @multilayer_perceptron_incremental_momentum,
      @multilayer_perceptron_incremental_adaptative_etha
    }
    {
      @multilayer_perceptron_batch
    }
  };
end

function withFunctionOutput = with_function(algorithm, improvement, layersAndSize, func, N, activationFunc, activationFuncDerived, minimumError, learningRate, betha, alpha, adaptativeA, adaptativeB, kEpochs)
  trainingSet = utils.get_training_set(func, N);
  testingSet{1} = [];
  testingSet{2} = [];

  networkWeights = test_network.configurations{algorithm}{improvement}(trainingSet, testingSet, false, 1, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha, alpha, adaptativeA, adaptativeB, kEpochs, adaptativeA, adaptativeB, kEpochs);

  for i = 1:rows(trainingSet{1})
    V = network_utils.forward_propagation([trainingSet{1}(i, :)], networkWeights, activationFunc, betha);
    trainingSet{1}(i, :)
    V{columns(layersAndSize)} >= 0.5
  end
end

function withTerrainOutput = with_terrain(filePath, trainingPercentage, algorithm, improvement, layersAndSize, activationFunc, activationFuncDerived, minimumError, learningRate, betha, alpha, adaptativeA, adaptativeB, kEpochs)
  dataSets = utils.get_random_subset(load(filePath), trainingPercentage);

  trainingSet = dataSets{1};
  normalizeTrainingInput = utils.normalize_x(dataSets{1}{2});
  trainingSet{2} = normalizeTrainingInput{1};

  if (trainingPercentage == 1)
    testingSet{1} = [];
    testingSet{2} = [];
  else
    testingSet = dataSets{2};
    normalizeTestingInput = utils.normalize_x(dataSets{2}{2});
    testingSet{2} = normalizeTestingInput{1};
  end

  networkWeights = test_network.configurations{algorithm}{improvement}(trainingSet, testingSet, true, trainingPercentage, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha, alpha, adaptativeA, adaptativeB, kEpochs);

  outputTrainingSet{1} = trainingSet{1};
  V = network_utils.forward_propagation(outputTrainingSet{1}, networkWeights, activationFunc, betha);
  outputTrainingSet{2} = V{columns(layersAndSize)};
  outputTrainingSet{2} = utils.denormalize_x(outputTrainingSet{2}, normalizeTrainingInput{2});

  outputTestingSet{1} = testingSet{1};
  V = network_utils.forward_propagation(outputTestingSet{1}, networkWeights, activationFunc, betha);
  outputTestingSet{2} = V{columns(layersAndSize)};
  outputTestingSet{2} = utils.denormalize_x(outputTestingSet{2}, normalizeTestingInput{2});

  outputSet{1} = [outputTrainingSet{1} ; outputTestingSet{1}];
  outputSet{2} = [outputTrainingSet{2} ; outputTestingSet{2}];

  fid = fopen('../../doc/data/result.txt', 'w+');
  fprintf(fid, '%d\n\n', rows(outputSet{1}));

  for i = 1:rows(outputSet{1})
    result = [outputSet{1}(i, :) outputSet{2}(i, :)];

    fprintf(fid, '%g ', result);
    fprintf(fid, '\n');
  end
  fclose(fid);
end
