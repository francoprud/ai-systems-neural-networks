% inputs: matrix that contains all inputs to analize with function
function output = parity(inputs)
  for i = 1:rows(inputs)
    result(i) = mod(sum(inputs(i, :)), 2) == 0;
  end

  output = result';
end
