% inputs: matrix that contains all inputs to analize with function
function output = simetry(inputs)
  inputSize = columns(inputs);
  for patternIndex = 1:rows(inputs)
    i = 1; j = inputSize;
    currentInput = inputs(patternIndex, :);
    result(patternIndex) = 0;
    while (i < j && !result(patternIndex))
      if (currentInput(i) != currentInput(j))
        result(patternIndex) = 1;
      end
      i++; j--;
    end
  end
  output = ~result';
end
