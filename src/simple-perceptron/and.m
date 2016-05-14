% inputs: matrix that contains all inputs to analize with function
function output = and(inputs)
  for i = 1:rows(inputs)
    result(i) = sum(inputs(i, :)) == columns(inputs);
  end

  output = result';
end
