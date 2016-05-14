function sigmoid = sigmoid()
  sigmoid.function = @sigmoid_func;
  sigmoid.eval = @sigmoid_eval;
end

% inputs: can be an array
function output = sigmoid_func(inputs)
  output = (1 + exp(-inputs)).**(-1);
end

% result: result from sigmoid_func
function outputEval = sigmoid_eval(result)
  outputEval = result >= 0.5;
end
