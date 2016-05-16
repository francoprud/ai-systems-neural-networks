% inputs: matrix that contains all inputs to analize with function
function output = xor(inputs)
  for i = 1:rows(inputs)
    result(i) = sum(inputs(i, :)) == 1;
  end

  output = result';
end
