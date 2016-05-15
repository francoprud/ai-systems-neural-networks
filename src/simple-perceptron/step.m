function step = step()
  step.function = @step_func;
  step.eval = @step_eval;
end

% inputs: can be an array
function step = step_func(inputs)
  step = inputs >= 0;
end

% result: result from step_func
function outputEval = step_eval(result)
  outputEval = result;
end
