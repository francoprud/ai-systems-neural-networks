%% utils: function description
function utils = utils()
  utils.get_training_set = @get_training_set;
end


function trainingSet = get_training_set(f, n)
  if (n <= 0)
    printf('Wrong n\n')
    return;
  end

  posiblesValues = get_posibles_values(n);
  trainingSet{1} = cartesian_product(posiblesValues);
  trainingSet{2} = feval(f, trainingSet{1});
end

% Returns a set of posibles values
% For example: {[0,1], [0,1], [0,1]}
function possibleValuesOutput = get_posibles_values(n)
  possibleValuesOutput = {};
  for i = 1:n
    possibleValuesOutput(i) = 0:1;
  end
end

% sets: set of vectors
function cartesianProductOutput = cartesian_product(sets)
  c = cell(1, numel(sets));
  [c{:}] = ndgrid(sets{:});
  cartesianProductOutput = cell2mat(cellfun(@(v)v(:), c, 'UniformOutput',false));
end
