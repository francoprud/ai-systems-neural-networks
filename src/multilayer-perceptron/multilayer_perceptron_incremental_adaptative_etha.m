% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
%   {1}: matrix NxM where N is the amount of inputs and M is the size of each input
%   {2}: matrix NxK where N is the amount of outputs (equal to inputs) and K is the size of each output
% minimumError: minimum error that determines when to stop training
% learningRate: positive parameter for gradient descent to work
% layersAndSize: array that represents amount of layers and neurons in each layer, starting from the first hidden layer
% activation_func:
% activation_func_derived:
% betha:
function output = multilayer_perceptron_incremental_adaptative_etha(trainingSet, testingSet, needsPlot, trainingPercentage, layersAndSize, minimumError, learningRate, activation_func, activation_func_derived, betha, alpha, adaptativeA, adaptativeB, kEpochs)
  totalInputs = rows(trainingSet{1});
  totalOutputs = rows(trainingSet{2});
  inputSize = columns(trainingSet{1});
  totalLayers = columns(layersAndSize) + 1; % All layers including entrance and output layers
  totalEdgesLayers = columns(layersAndSize); % Total amount of edges "spaces"
  epoch = 0;
  counter = 0;
  lastTrainingError = [];
  lastTestingError = [];
  lastLearningRate = [];

  if (totalInputs != totalOutputs)
    printf('The training set is invalid.\n');
    return;
  end

  if (needsPlot)
    utils.plot_original_function(trainingSet, testingSet);
    drawnow;
  end

  networkWeights = network_utils.randomize_network_weights([inputSize layersAndSize]);
  V = network_utils.forward_propagation(trainingSet{1}, networkWeights, activation_func, betha); % Contains the biases of hidden layers and output layers
  currentError = network_utils.calculate_error(trainingSet{2}, V{totalEdgesLayers});

  currentMinimumError = currentError;
  currentMinimumTestError = 1000;

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

      previousNetworkWeights = networkWeights;

      for j = (totalEdgesLayers):-1:2
        layers = networkWeights{j}(2 : end, :); % Remove -1 neuron to layers
        delta{j - 1} = activation_func_derived(V{j - 1}, betha).*(delta{j} * layers');
        deltaWeight = learningRate * [-1 V{j - 1}]' * delta{j};
        networkWeights{j} = networkWeights{j} + deltaWeight;
      end
      deltaWeight = learningRate * [-1 trainingSet{1}(currentIndex, :)]' * delta{1};
      networkWeights{1} = networkWeights{1} + deltaWeight;
    end

    V = network_utils.forward_propagation(trainingSet{1}, networkWeights, activation_func, betha);

    previousError = currentError;
    currentError = network_utils.calculate_error(trainingSet{2}, V{totalLayers -1});

    if (currentError < currentMinimumError)
      currentMinimumError = currentError;
    end

    epoch++;

    if (currentError - previousError < 0)
      counter++;
      if (counter >= kEpochs)
        learningRate += adaptativeA;
      end
    else
      counter = 0;
      learningRate *= (1 - adaptativeB);
      currentError = previousError;
      networkWeights = previousNetworkWeights;
    end

    if (mod(epoch, utils.step) == 0)
      percentageLearned = utils.calculate_errors(trainingSet, testingSet, networkWeights, activation_func, betha);

      if (needsPlot)
        utils.plot_training_set(trainingSet{1}, V{totalLayers - 1});

        if (trainingPercentage != 1)
          utils.plot_testing_set(testingSet, networkWeights, activation_func, betha);
          testError = network_utils.get_test_error(networkWeights, testingSet, activation_func, betha);
        else
          testError = 0;
        end

        if (currentMinimumTestError > testError)
          currentMinimumTestError = testError;
        end

        trainingErrors = [lastTrainingError currentError];
        testingErrors = [lastTestingError testError];
        utils.plot_error_vs_epoch(epoch, trainingErrors, testingErrors);
        lastTrainingError = [currentError];
        lastTestingError = [testError];

        learningRates = [lastLearningRate learningRate];
        utils.plot_learning_rate_vs_epoch(epoch, learningRates);
        lastLearningRate = [learningRate];

        utils.plot_aproximated_function(networkWeights, trainingSet, testingSet, activation_func, betha, totalEdgesLayers);
        drawnow;

        printf('epocas = %d; currentError = %g; currentMinimumError = %g; testError = %g; currentMinimumTestError = %g; percentageLearned = %g\n', epoch, currentError, currentMinimumError, testError, currentMinimumTestError, percentageLearned);
      else
        printf('epocas = %d; currentError = %g; currentMinimumError = %g; percentageLearned = %g\n', epoch, currentError, currentMinimumError, percentageLearned);
      end

      fflush(stdout);
    end
  end

  printf('epocas = %d; currentError = %g; currentMinimumError = %g\n', epoch, currentError, currentMinimumError);
  fflush(stdout);

  output = networkWeights;
end
