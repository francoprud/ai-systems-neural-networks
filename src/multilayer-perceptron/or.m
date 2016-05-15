% inputs: matrix that contains all inputs to analize with function
function output = or(inputs)
  for i = 1:rows(inputs)
    result(i) = sum(inputs(i, :)) > 0;
  end

  output = result';
end
