%% utils: function description
function utils = utils()
  utils.get_training_set = @get_training_set;
  utils.get_random_subset = @get_random_subset;

  utils.get_activation_funs = @get_activation_funs;
  utils.sizeFactor = 15;
  utils.step = 1;
  utils.fontSize = 15;
  utils.markersize = 10;
  utils.linewidth = 3;
  utils.plot_original_function = @plot_original_function;
  utils.plot_training_set = @plot_training_set;
  utils.plot_error_vs_epoch = @plot_error_vs_epoch;
  utils.plot_learning_rate_vs_epoch = @plot_learning_rate_vs_epoch;
  utils.plot_aproximated_function = @plot_aproximated_function;
  utils.plot_testing_set = @plot_testing_set;

  utils.normalize_x = @normalize_x;
  utils.denormalize_x = @denormalize_x;

  utils.calculate_errors = @calculate_errors;
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

function plot_original_function(trainingSet, testingSet, activationFunsId)
  figure(1);

  inputs = [trainingSet{1} ; testingSet{1}];
  outputs = [trainingSet{2} ; testingSet{2}];
  
  plot3(inputs(:,1), inputs(:,2), outputs, '.b', 'markersize', utils.markersize)

  switch activationFunsId
    case 1
      % Tangente hiperbolica
      axis([-1 1 -1 1 -1 1]);
    case 2
      % Exponencial
      axis([-1 1 -1 1 0 1]);
  end
  title('Original function', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_training_set(trainingSet, networkOutputs, activationFunsId)
  figure(2);

  plot3(trainingSet{1}(:,1), trainingSet{1}(:,2), networkOutputs, '.r', 'markersize', utils.markersize)

  switch activationFunsId
    case 1
      % Tangente hiperbolica
      axis([-1 1 -1 1 -1 1]);
    case 2
      % Exponencial
      axis([-1 1 -1 1 0 1]);
  end
  title('Patterns (X, Y) vs learned Z', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_testing_set(testingSet, networkWeights, activation_func, betha)
  if (rows(testingSet{1}) == 0)
    return;
  end

  figure(3);

  inputs = testingSet{1};
  outputs = network_utils.forward_propagation(inputs, networkWeights, activationFunsId, activation_func, betha);
  netOutputs = outputs{columns(outputs)};

  plot3(inputs(:,1), inputs(:,2), netOutputs, '.r', 'markersize', utils.markersize)

  switch activationFunsId
    case 1
      % Tangente hiperbolica
      axis([-1 1 -1 1 -1 1]);
    case 2
      % Exponencial
      axis([-1 1 -1 1 0 1]);
  end
  title('Testing patterns (X, Y) vs learned Z', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_error_vs_epoch(epoch, trainingErrors, testingErrors, plotTestError)
  figure(4);

  if (plotTestError)
    plot(1:epoch, trainingErrors, '-k', 'linewidth', utils.linewidth, 'markersize', utils.markersize, 1:epoch, testingErrors, '-r', 'linewidth', utils.linewidth, 'markersize', utils.markersize)
    title('Training error and testing error', 'fontsize', utils.fontSize);
  else
    plot(1:epoch, trainingErrors, '-k', 'linewidth', utils.linewidth, 'markersize', utils.markersize)
    title('Training error', 'fontsize', utils.fontSize);
  end

  xlabel('Epoch', 'fontsize', utils.fontSize);
  ylabel('Error', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize)
end

function plot_learning_rate_vs_epoch(epoch, learningRates)
  figure(5);

  plot(1:epoch, learningRates, '-k', 'linewidth', utils.linewidth, 'markersize', utils.markersize)

  title('Learning rate vs epoch', 'fontsize', utils.fontSize);
  xlabel('Epoch', 'fontsize', utils.fontSize);
  ylabel('Learning rate', 'fontsize', utils.fontSize);
  set(gca, 'fontsize', utils.fontSize);
end

function plot_aproximated_function(networkWeights, trainingSet, testingSet, activationFunsId, activation_func, beta)
  figure(6);

  inputs = [trainingSet{1} ; testingSet{1}];
  V = network_utils.forward_propagation(inputs, networkWeights, activation_func, beta);
  networkOutputs = V{end};
  originalOutputs = [trainingSet{2} ; testingSet{2}];

  hold off;
  plot3(inputs(:,1), inputs(:,2), networkOutputs, '.r', 'markersize', utils.markersize) % Learned function
  hold on;
  plot3(inputs(:,1), inputs(:,2), originalOutputs, '.b', 'markersize', utils.markersize); % Original function

  switch activationFunsId
    case 1
      % Tangente hiperbolica
      axis([-1 1 -1 1 -1 1]);
    case 2
      % Exponencial
      axis([-1 1 -1 1 0 1]);
  end
  legend('Learned function', 'Original function');
  title('Learned function', 'fontsize', utils.fontSize);
  xlabel('X', 'fontsize', utils.fontSize);
  ylabel('Y', 'fontsize', utils.fontSize);
  zlabel('Z', 'fontsize', utils.fontSize);
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
      delta = 0.1;
    case 2
      % Exponencial
      delta = 0.05;
  end
  inputs = [trainingSet{1}; testingSet{1}];
  outputs = network_utils.forward_propagation(inputs, networkWeights, activation_func, betha);
  expectedOutputs = [trainingSet{2}; testingSet{2}];

  result = abs(expectedOutputs - outputs{columns(networkWeights)}) <= delta;

  calculateErrorsOutput = sum(result) / rows(inputs);
end
