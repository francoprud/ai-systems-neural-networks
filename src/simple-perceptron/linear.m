function linear = linear()
  linear.function = @linear_func;
  linear.eval = @linear_eval;
end

% inputs: can be an array
function output = linear_func(inputs)
  output = inputs;
end

% result: result from linear_func
function outputEval = linear_eval(result)
  outputEval = result >= 0;
end
