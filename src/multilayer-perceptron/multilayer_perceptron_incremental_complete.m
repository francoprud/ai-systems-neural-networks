% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
%   {1}: matrix NxM where N is the amount of inputs and M is the size of each input
%   {2}: matrix NxK where N is the amount of outputs (equal to inputs) and K is the size of each output
% minimumError: minimum error that determines when to stop training
% learningRate: positive parameter for gradient descent to work
% layersAndSize: array that represents amount of layers and neurons in each layer, starting from the first hidden layer
% activation_func:
% activation_func_derived:
% betha:
function output = multilayer_perceptron_incremental_complete(trainingSet, testingSet, needsPlot, trainingPercentage, layersAndSize, minimumError, learningRate, activationFunsId, activation_func, activation_func_derived, betha, alpha, adaptativeA, adaptativeB, kEpochs)
  totalInputs = rows(trainingSet{1});
  totalOutputs = rows(trainingSet{2});
  inputSize = columns(trainingSet{1});
  totalLayers = columns(layersAndSize) + 1; % All layers including entrance and output layers
  totalEdgesLayers = columns(layersAndSize); % Total amount of edges "spaces"
  alphaValue = alpha;
  epoch = 0;
  counter = 0;
  trainingErrors = [];
  testingErrors = [];
  learningRates = [];

  if (totalInputs != totalOutputs)
    printf('The training set is invalid.\n');
    return;
  end

  if (needsPlot)
    utils.plot_original_function(trainingSet, testingSet, activationFunsId);
  end

  networkWeights = network_utils.randomize_network_weights([inputSize layersAndSize]);

  for i = 1:columns(networkWeights)
    previousDeltaWeight{i} = zeros(rows(networkWeights{i}), columns(networkWeights{i}));
  end

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
        deltaWeight = learningRate * [-1 V{j - 1}]' * delta{j} + alphaValue * previousDeltaWeight{j};
        previousDeltaWeight{j} = deltaWeight;
        networkWeights{j} = networkWeights{j} + deltaWeight;
      end
      deltaWeight = learningRate * [-1 trainingSet{1}(currentIndex, :)]' * delta{1} + alphaValue * previousDeltaWeight{1};
      previousDeltaWeight{1} = deltaWeight;
      networkWeights{1} = networkWeights{1} + deltaWeight;
    end

    V = network_utils.forward_propagation(trainingSet{1}, networkWeights, activation_func, betha);

    previousError = currentError;
    currentError = network_utils.calculate_error(trainingSet{2}, V{totalLayers - 1});

    % Updating minimum training error
    if (currentError < currentMinimumError)
      currentMinimumError = currentError;
    end

    % Calculating test error
    if (trainingPercentage != 1)
      testError = network_utils.get_test_error(networkWeights, testingSet, activation_func, betha);
    else
      testError = 0;
    end

    % Updating minimum test error
    if (currentMinimumTestError > testError)
      currentMinimumTestError = testError;
    end

    if (adaptativeA != 0 || adaptativeB != 0)
      if (currentError - previousError < 0)
        alphaValue = alpha;
        counter++;
        if (counter >= kEpochs)
          previousError
          currentError
          learningRate
          learningRate += adaptativeA;
          learningRate
          epoch
        end
      else
        alphaValue = 0;
        counter = 0;
        learningRate *= (1 - adaptativeB);
        currentError = previousError;
        networkWeights = previousNetworkWeights;
      end
    end

    epoch++;
    learningRates(end+1) = learningRate;
    trainingErrors(end+1) = currentError;
    testingErrors(end+1) = testError;

    % Plots
    if (mod(epoch, utils.step) == 0 || currentError <= minimumError)
      percentageLearned = utils.calculate_errors(trainingSet, testingSet, networkWeights, activationFunsId, activation_func, betha);

      if (needsPlot)
        utils.plot_training_set(trainingSet, V{end}, activationFunsId);
        utils.plot_testing_set(testingSet, networkWeights, activationFunsId, activation_func, betha);
        plotTestError = !(trainingPercentage == 1);
        utils.plot_error_vs_epoch(epoch, trainingErrors, testingErrors, plotTestError);
        utils.plot_learning_rate_vs_epoch(epoch, learningRates);
        utils.plot_aproximated_function(networkWeights, trainingSet, testingSet, activationFunsId, activation_func, betha);
        drawnow;
      end
      
      printf('epocas = %d; currentError = %g; currentMinimumError = %g; testError = %g; currentMinimumTestError = %g; percentageLearned = %g\n', epoch, currentError, currentMinimumError, testError, currentMinimumTestError, percentageLearned);
      fflush(stdout);
    end  
  end

  output = networkWeights;

end
