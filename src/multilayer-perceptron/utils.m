%% utils: function description
function utils = utils()
  utils.get_training_set = @get_training_set;
  utils.get_random_subset = @get_random_subset;

  utils.sizeFactor = 15;
  utils.step = 20;
  utils.plot_original_function = @plot_original_function;
  utils.plot_training_set = @plot_training_set;
  utils.plot_error_vs_epoch = @plot_error_vs_epoch;
  utils.plot_learning_rate_vs_epoch = @plot_learning_rate_vs_epoch;
  utils.plot_aproximated_function = @plot_aproximated_function;
  utils.plot_testing_set = @plot_testing_set;

  utils.normalize_x = @normalize_x;
  utils.denormalize_x = @denormalize_x;
end

function trainingSet = get_training_set(f, n)
  if (n <= 0)
    printf('Wrong n\n')
    return;
  end

  posiblesValues = get_posibles_values(n);
  trainingSet{1} = cartesian_product(posiblesValues);
  trainingSet{2} = feval(f, trainingSet{1});
end

% Returns a set of posibles values
% For example: {[0,1], [0,1], [0,1]}
function possibleValuesOutput = get_posibles_values(n)
  possibleValuesOutput = {};
  for i = 1:n
    possibleValuesOutput(i) = 0:1;
  end
end

% sets: set of vectors
function cartesianProductOutput = cartesian_product(sets)
  c = cell(1, numel(sets));
  [c{:}] = ndgrid(sets{:});
  cartesianProductOutput = cell2mat(cellfun(@(v)v(:), c, 'UniformOutput',false));
end

% matrix: The matrix to get random rows
% percentage: The percentage of the matrix to get random
function randomSubsetOutput = get_random_subset(matrix, percentage)
  matrixRows = rows(matrix);
  rowsOrder = randperm(matrixRows);
  mustExtract = floor(percentage * matrixRows);

  randomSubsetOutput{1}{1} = matrix(rowsOrder(1:mustExtract), 1:2);
  randomSubsetOutput{1}{2} = matrix(rowsOrder(1:mustExtract), 3);

  randomSubsetOutput{2}{1} = matrix(rowsOrder((mustExtract + 1):matrixRows), 1:2);
  randomSubsetOutput{2}{2} = matrix(rowsOrder((mustExtract + 1):matrixRows), 3);
end

function plot_original_function(trainingSet, testingSet)
  subplot(2,3,1);

  inputs = [trainingSet{1} ; testingSet{1}];
  outputs = [trainingSet{2} ; testingSet{2}];
  s = ones(length(outputs), 1) .* utils.sizeFactor; %size
  c = outputs; % Color relative to height

  scatter3(inputs(:,1), inputs(:,2), outputs, s, c, 'filled')
  axis([-3 3.5 -3 3 -1.1 1.1]);

  title('Original function');
  xlabel('X');
  ylabel('Y');
  zlabel('Z');
end

function plot_training_set(trainingSet, outputs)
  subplot(2,3,2);

  s = ones(length(outputs), 1) .* utils.sizeFactor; %size
  c = outputs;

  scatter3(trainingSet(:,1), trainingSet(:,2), outputs, s, c, 'filled')
  axis([-3 3.5 -3 3 -1.1 1.1]);

  title('TrainingSet vs outputs');
  xlabel('X');
  ylabel('Y');
  zlabel('Z');end

function plot_testing_set(testingSet, networkWeights, activation_func, betha)
  subplot(2,3,3);

  inputs = testingSet{1};
  outputs = network_utils.forward_propagation(inputs, networkWeights, activation_func, betha);
  netOutputs = outputs{columns(outputs)};
  s = ones(length(netOutputs), 1) .* utils.sizeFactor; %size
  c = netOutputs;

  scatter3(inputs(:,1), inputs(:,2), netOutputs, s, c, 'filled')
  axis([-3 3.5 -3 3 -1.1 1.1]);

  title('TestingSet vs outputs');
  xlabel('X');
  ylabel('Y');
  zlabel('Z');
end

function plot_error_vs_epoch(epoch, trainingErrors, testingErrors)
  subplot(2,3,4)
  hold on;
  
  plot([epoch-utils.step epoch], trainingErrors, '-ok', [epoch-utils.step epoch], testingErrors, '-or')

  title('TrainingError and testingEror');
  xlabel('Epoch');
  ylabel('Error');
end

function plot_learning_rate_vs_epoch(epoch, learningRates)
  subplot(2,3,5)
  hold on;

  plot([epoch-utils.step epoch], learningRates, '-ok')

  title('LearningRate vs epoch');
  xlabel('Epoch');
  ylabel('LearningRate');
end

function plot_aproximated_function(networkWeights, trainingSet, testingSet, activation_func, betha, totalLayers)
  subplot(2,3,6)

  inputs = [trainingSet{1} ; testingSet{1}];
  outputs = network_utils.forward_propagation(inputs, networkWeights, activation_func, betha);
  netOutputs = outputs{columns(outputs)};

  s = ones(length(netOutputs), 1) .* utils.sizeFactor; %size
  c = netOutputs; % Color relative to height

  scatter3(inputs(:,1), inputs(:,2), netOutputs, s, c, 'filled');
  axis([-3 3.5 -3 3 -1.1 1.1]);

  title('Aproximated function');
  xlabel('X');
  ylabel('Y');
  zlabel('Z');
end

function output = normalize_x(A)
  maximum = max(A);
  minimum = min(A);
  range = maximum - minimum;
  output{2}(1) = minimum + (range/2);
  output{2}(2) = range/2;
  output{1} = (A.-(output{2}(1)))./output{2}(2);
end

function denormalizedOutput = denormalize_x(A, B)
  denormalizedOutput = A.*(B(2));
  for i = 1:columns(A)
    denormalizedOutput(i) = denormalizedOutput(i) + B(1);
  end
end
