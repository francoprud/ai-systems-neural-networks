% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
% minimumError: minimum error that determines when to stop training
% learingRate: positive parameter for gradient descent to work
% func: activation function (logistic function/transference function)
function network = neural_network(trainingSet, minimumError, learningRate, func)
  totalInputs = rows(trainingSet{1});
  totalOutputs = rows(trainingSet{2});
  inputSize = columns(trainingSet{1});

  if (totalInputs != totalOutputs)
    printf('The training set is invalid.\n');
    return;
  end

  W = rand(inputSize + 1, 1).*2.-1; % Generates random values between [-1, 1]
  currentError = 1;

  while (currentError > minimumError)
    inputsOrder = randperm(totalInputs); % Randomize the order of inputs

    for i = 1:totalInputs
      output = [-1 trainingSet{1}(inputsOrder(i), :)] * W;
      difference = trainingSet{2}(inputsOrder(i), :) - func(output);

      deltaW = (learningRate * difference) * [-1 trainingSet{1}(inputsOrder(i), :)]';
      W = W + deltaW;
    end

    currentError = calculateError(W, trainingSet, func);
  end
  network = W;
end

% W: array of weights
% trainingSet: cell array. In {1} has matrix with the inputs. In {2} has expected result of inputs from {1}
% func: activation function (logistic function/transference function)
function err = calculateError(W, trainingSet, func)
  totalInputs = rows(trainingSet{1});
  allInputs = [ones(totalInputs, 1).*(-1) trainingSet{1}];
  allOutputs = allInputs * W;
  outputsDifference = trainingSet{2} - func(allOutputs);

  err = sum(outputsDifference.**2) / (2 * totalInputs);
end
