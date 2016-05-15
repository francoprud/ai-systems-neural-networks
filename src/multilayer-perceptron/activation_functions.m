function activationFunctions = activation_functions()
  activationFunctions.tanh = @tanh_func;
  activationFunctions.tanh_derived = @tanh_derived_func;

  activationFunctions.exp = @exp_func;
  activationFunctions.exp_derived = @exp_derived_func;

  activationFunctions.sigmoid = @sigmoid_func;
  activationFunctions.sigmoid_eval = @sigmoid_eval;

  activationFunctions.step = @step_func;
  activationFunctions.step_eval = @step_eval;

  activationFunctions.linear = @linear_func;
  activationFunctions.linear_eval = @linear_eval;
end

% -------- tanh functions -------- %
function tanhOutput = tanh_func(h, betha)
  tanhOutput = tanh(h * betha);
end

function tanhDerivedOutput = tanh_derived_func(h, betha)
  tanhDerivedOutput = betha * (1 - h.**2);
end

% -------- exponential functions -------- %
function expOutput = exp_func(h, betha)
  expOutput = (1 + exp(-2 * h.*betha)).**(-1);
end

function expDerivedOutput = exp_derived_func(h, betha)
  expDerivedOutput = 2 * betha * h.*(1 - h);
end

% -------- sigmoid functions -------- %
function sigmoidOutput = sigmoid_func(inputs, betha)
  output = (1 + exp(-inputs)).**(-1);
end

function sigmoidEval = sigmoid_eval(result)
  outputEval = result >= 0.5;
end

% -------- step functions -------- %
function stepOutput = step_func(inputs, betha)
  step = inputs >= 0;
end

function stepEval = step_eval(result)
  outputEval = result;
end

% -------- linear functions -------- %
function linearOutput = linear_func(inputs, betha)
  output = inputs;
end

function linearEval = linear_eval(result)
  outputEval = result >= 0;
end
