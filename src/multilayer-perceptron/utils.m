%% utils: function description
function utils = utils()
  utils.get_training_set = @get_training_set;
  utils.get_random_subset = @get_random_subset;

  utils.get_axis_values = @get_axis_values;
  utils.get_activation_funs = @get_activation_funs;
  utils.sizeFactor = 15;
  utils.step = 100;
  utils.fontSize = 20;
  utils.markersize = 10;
  utils.linewidth = 3;
  utils.plot_original_function = @plot_original_function;
  utils.plot_training_set = @plot_training_set;
  utils.plot_error_vs_epoch = @plot_error_vs_epoch;
  utils.plot_learning_rate_vs_epoch = @plot_learning_rate_vs_epoch;
  utils.plot_aproximated_function = @plot_aproximated_function;
  utils.plot_testing_set = @plot_testing_set;
  utils.plot_percentagesLearned = @plot_percentagesLearned;

  utils.normalize_x = @normalize_x;
  utils.denormalize_x = @denormalize_x;

  utils.calculate_errors = @calculate_errors;
end

function axisOutput = get_axis_values(dataset)
  totalSet{1} = [dataset{1}{1}; dataset{2}{1}];
  totalSet{2} = [dataset{1}{2}; dataset{2}{2}];

  minX = min(totalSet{1}(:, 1));
  maxX = max(totalSet{1}(:, 1));

  minY = min(totalSet{1}(:, 2));
  maxY = max(totalSet{1}(:, 2));

  minZ = min(totalSet{2});
  maxZ = max(totalSet{2});

  axisOutput = [floor(minX) ceil(maxX) floor(minY) ceil(maxY) floor(minZ) ceil(maxZ)];
end

function funs = get_activation_funs(x)
  switch(x)
  case 1
    %tangente hiperbolica
    funs{1} = activation_functions.tanh;
    funs{2} = activation_functions.tanh_derived;
  case 2
    %exponencial
    funs{1} = activation_functions.exp;
    funs{2} = activation_functions.exp_derived;
  end
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

function plot_original_function(trainingSet, testingSet, axisValues)
  figure(1);

  inputs = [trainingSet{1} ; testingSet{1}];
  outputs = [trainingSet{2} ; testingSet{2}];
  
  plot3(inputs(:,1), inputs(:,2), outputs, '.b', 'markersize', utils.markersize)
  axis(axisValues);
  title('Original function', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_training_set(trainingSet, networkOutputs, axisValues)
  figure(2);

  plot3(trainingSet{1}(:,1), trainingSet{1}(:,2), networkOutputs, '.r', 'markersize', utils.markersize)
  axis(axisValues);
  title('Patterns (X, Y) vs learned Z', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_testing_set(testingSet, networkWeights, axisValues, activation_func, betha)
  if (rows(testingSet{1}) == 0)
    return;
  end

  figure(3);

  inputs = testingSet{1};
  outputs = network_utils.forward_propagation(inputs, networkWeights, activation_func, betha);
  netOutputs = outputs{columns(outputs)};

  plot3(inputs(:,1), inputs(:,2), netOutputs, '.r', 'markersize', utils.markersize)
  axis(axisValues);
  title('Testing patterns (X, Y) vs learned Z', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_error_vs_epoch(epoch, trainingErrors, testingErrors, plotTestError)
  figure(4);

  if (plotTestError)
    plot(1:epoch, trainingErrors, '-k', 'linewidth', utils.linewidth, 1:epoch, testingErrors, '-r', 'linewidth', utils.linewidth)
    title('Training error and testing error', 'fontsize', utils.fontSize);
  else
    plot(1:epoch, trainingErrors, '-k', 'linewidth', utils.linewidth)
    title('Training error', 'fontsize', utils.fontSize);
  end

  xlabel('Epoch', 'fontsize', utils.fontSize);
  ylabel('Error', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize)
end

function plot_learning_rate_vs_epoch(epoch, learningRates)
  figure(5);

  plot(1:epoch, learningRates, '-k', 'linewidth', utils.linewidth)

  title('Learning rate vs epoch', 'fontsize', utils.fontSize);
  xlabel('Epoch', 'fontsize', utils.fontSize);
  ylabel('Learning rate', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_aproximated_function(networkWeights, trainingSet, testingSet, axisValues, activation_func, beta)
  figure(6);

  inputs = [trainingSet{1} ; testingSet{1}];
  V = network_utils.forward_propagation(inputs, networkWeights, activation_func, beta);
  networkOutputs = V{end};
  originalOutputs = [trainingSet{2} ; testingSet{2}];

  hold off;
  plot3(inputs(:,1), inputs(:,2), networkOutputs, '.r', 'markersize', utils.markersize) % Learned function
  hold on;
  plot3(inputs(:,1), inputs(:,2), originalOutputs, '.b', 'markersize', utils.markersize); % Original function
  axis(axisValues);
  legend('Learned function', 'Original function');
  title('Learned function', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_percentagesLearned(percentagesLearned, epoch)
  figure(7);

  plot(1:epoch, percentagesLearned, '-k')

  title('Percentages learned vs epoch', 'fontsize', utils.fontSize);
  xlabel('Epoch', 'fontsize', utils.fontSize);
  ylabel('Percentage learned', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function output = normalize_x(A, a, b)
  maximum = max(A);
  minimum = min(A);
  range = maximum - minimum;
  radius = (b - a)/2;
  output{2}(1) = minimum + (range/2);
  output{2}(2) = range/(2*radius);
  output{2}(3) = radius + a;
  output{1} = ((A.-(output{2}(1)))./output{2}(2)).+output{2}(3);
end

function denormalizedOutput = denormalize_x(A, B)
  if (rows(A) == 0)
    denormalizedOutput = [];
    return;
  end
  transpose_after = false;
  Ap = A;
  if (columns(A) == 1)
    transpose_after = true;
    Ap = A';
  end
  for i = 1:columns(Ap)
    denormalizedOutput(i) = Ap(i) - B(3);
  end
  denormalizedOutput = denormalizedOutput.*(B(2));
  for i = 1:columns(Ap)
    denormalizedOutput(i) = denormalizedOutput(i) + B(1);
  end
  if (transpose_after)
    denormalizedOutput = denormalizedOutput';
  end
end

function calculateErrorsOutput = calculate_errors(trainingSet, testingSet, networkWeights, activationFunsId, activation_func, betha)
  switch activationFunsId
    case 1
      % Tangente hiperbolica
      delta = 0.02;
    case 2
      % Exponencial
      delta = 0.01;
  end
  inputs = [trainingSet{1}; testingSet{1}];
  outputs = network_utils.forward_propagation(inputs, networkWeights, activation_func, betha);
  expectedOutputs = [trainingSet{2}; testingSet{2}];

  result = abs(expectedOutputs - outputs{columns(networkWeights)}) <= delta;

  calculateErrorsOutput = sum(result) / rows(inputs);
end
