function testNetwork = test_network()
  testNetwork.with_function = @with_function;
  testNetwork.with_terrain = @with_terrain;
end

function withFunctionOutput = with_function(layersAndSize, func, N, activationFunc, activationFuncDerived, minimumError, learningRate, betha, alpha, adaptativeA, adaptativeB, kEpochs)
  trainingSet = utils.get_training_set(func, N);
  testingSet{1} = [];
  testingSet{2} = [];

  networkWeights = multilayer_perceptron_incremental_complete(trainingSet, testingSet, false, 1, layersAndSize, minimumError, learningRate, activationFunc, activationFuncDerived, betha, alpha, adaptativeA, adaptativeB, kEpochs, adaptativeA, adaptativeB, kEpochs);

  for i = 1:rows(trainingSet{1})
    V = network_utils.forward_propagation([trainingSet{1}(i, :)], networkWeights, activationFunc, betha);
    trainingSet{1}(i, :)
    V{columns(layersAndSize)} >= 0.5
  end
end

%{
  activationFunsId es 1 si usamos tangente hiperbolica o 2 si usamos la exponencial.
  todos los patrones de entrada se normalizan entre patternsMinNorm y patternsMaxNorm.
  todo el output se normaliza entre outputMinNorm y outputMaxNorm.
%}
function withTerrainOutput = with_terrain(filePath, trainingPercentage, layersAndSize, activationFunsId, minimumError, learningRate, betha, alpha, adaptativeA, adaptativeB, kEpochs, needPlots)
  activationFuns = utils.get_activation_funs(activationFunsId);
  activationFunc = activationFuns{1};
  activationFuncDerived = activationFuns{2};
  patternsMinNorm = -1;
  patternsMaxNorm = 1;
  outputMinNorm = -1;
  outputMaxNorm = 1;
  switch(activationFunsId)
  case 1
    % Tangente hiperbolica
    outputMinNorm = -0.9;
    outputMaxNorm = 0.9;
  case 2
    % Exponencial
    outputMinNorm = 0.1;
    outputMaxNorm = 0.9;
  end

  dataSets = utils.get_random_subset(load(filePath), trainingPercentage);

  % e.g: trainingSet{1} are the training patterns & trainingSet{2} the training output
  trainingSet = dataSets{1};
  testingSet = dataSets{2};

  % trainingSetNormalizedWithDenormalizationVectors{1}{1} -> patterns del trainingSet normalizados
  % trainingSetNormalizedWithDenormalizationVectors{1}{2} -> vector de desnormalizacion de los patterns de trainingSet
  % trainingSetNormalizedWithDenormalizationVectors{2}{1} -> output del trainingSet normalizado
  % trainingSetNormalizedWithDenormalizationVectors{2}{2} -> vector de desnormalizacion del output de trainingSet
  trainingSetPatternsXNormalized = utils.normalize_x(trainingSet{1}(:, 1), patternsMinNorm, patternsMaxNorm){1};
  trainingSetPatternsYNormalized = utils.normalize_x(trainingSet{1}(:, 2), patternsMinNorm, patternsMaxNorm){1};
  trainingSetNormalizedWithDenormalizationVectors{1}{1} = [trainingSetPatternsXNormalized trainingSetPatternsYNormalized];

  % Won't define trainingSetNormalizedWithDenormalizationVectors{1}{2} because it's not necessary
  trainingSetNormalizedWithDenormalizationVectors{2} = utils.normalize_x(trainingSet{2}, outputMinNorm, outputMaxNorm);

  % the following commented line is to have denormalized patterns
  %trainingSetNormalized{1} = trainingSetNormalizedWithDenormalizationVectors{1}{1};
  trainingSetNormalized{1} = trainingSet{1};
  trainingSetNormalized{2} = trainingSetNormalizedWithDenormalizationVectors{2}{1};

  % This line is just to declare it. it will have a different value later. not sure if it's necessary.
  testingSetNormalizedWithDenormalizationVectors = trainingSetNormalizedWithDenormalizationVectors;

  % normalizeTrainingInput = utils.normalize_x(dataSets{1}{2});
  % trainingSet{2} = normalizeTrainingInput{1};

  % This if,else statement is needed because the case trainingPercentage=1 is special.
  % If we run the else statement everytime, it would fail.
  if (trainingPercentage == 1)
    testingSetNormalizedWithDenormalizationVectors{1}{1} = [];
    testingSetNormalizedWithDenormalizationVectors{2}{1} = [];
  else
    testingSetPatternsXNormalized = utils.normalize_x(testingSet{1}(:, 1), patternsMinNorm, patternsMaxNorm){1};
    testingSetPatternsYNormalized = utils.normalize_x(testingSet{1}(:, 2), patternsMinNorm, patternsMaxNorm){1};

    testingSetNormalizedWithDenormalizationVectors{1}{1} = [testingSetPatternsXNormalized testingSetPatternsYNormalized];
    % Won't define testingSetNormalizedWithDenormalizationVectors{1}{2} because it's not necessary
    % testingSetNormalizedWithDenormalizationVectors{1} = utils.normalize_x(testingSet{1});
    testingSetNormalizedWithDenormalizationVectors{2} = utils.normalize_x(testingSet{2}, outputMinNorm, outputMaxNorm);
  end

  % This works just like trainingSetNormalized
  % the following commented line is to have denormalized patterns
  %testingSetNormalized{1} = testingSetNormalizedWithDenormalizationVectors{1}{1};
  testingSetNormalized{1} = testingSet{1};
  testingSetNormalized{2} = testingSetNormalizedWithDenormalizationVectors{2}{1};

  dataSetsNormalized{1} = trainingSetNormalized;
  dataSetsNormalized{2} = testingSetNormalized;

  networkWeights = multilayer_perceptron_incremental_complete(trainingSetNormalized, testingSetNormalized, utils.get_axis_values(dataSetsNormalized), needPlots, trainingPercentage, layersAndSize, minimumError, learningRate, activationFunsId, activationFunc, activationFuncDerived, betha, alpha, adaptativeA, adaptativeB, kEpochs);

  % No hace falta desnormalizar la entrada porque ya la tengo desnormalizada en trainingSet
  % para la salida necesito recuperar el vector de desnormalizacion
  outputTrainingSet{1} = trainingSet{1};
  V = network_utils.forward_propagation(trainingSetNormalized{1}, networkWeights, activationFunc, betha);
  outputTrainingSet{2} = V{columns(layersAndSize)};
  outputTrainingSet{2} = utils.denormalize_x(outputTrainingSet{2}, trainingSetNormalizedWithDenormalizationVectors{2}{2});

  outputTestingSet{1} = testingSet{1};
  V = network_utils.forward_propagation(testingSetNormalized{1}, networkWeights, activationFunc, betha);
  outputTestingSet{2} = V{columns(layersAndSize)};
  outputTestingSet{2} = utils.denormalize_x(outputTestingSet{2}, testingSetNormalizedWithDenormalizationVectors{2}{2});

  outputSet{1} = [outputTrainingSet{1} ; outputTestingSet{1}];
  outputSet{2} = [outputTrainingSet{2} ; outputTestingSet{2}];

  fid = fopen('../../doc/data/result.txt', 'w+');
  fprintf(fid, '%d\n\n', rows(outputSet{1}) * 2);

  utils.print_ovito_file(fid, outputSet, '1 0 0');

  totalSet{1} = [dataSets{1}{1}; dataSets{2}{1}];
  totalSet{2} = [dataSets{1}{2}; dataSets{2}{2}];
  utils.print_ovito_file(fid, totalSet, '0 0 1');

  fclose(fid);

  withTerrainOutput = networkWeights
  
end
