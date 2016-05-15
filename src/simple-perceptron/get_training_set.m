% f: Function to apply
% n: Amount of bits to generate
function t = get_training_set(f, n)
	if (n <= 0)
		printf('Wrong n\n')
		return;
	end

	posiblesValues = get_posiblesValues(n);
	t{1} = cartesian_product(posiblesValues);
	t{2} = feval(f, t{1});

end

% Returns a set of posibles values
% For example: {[0,1], [0,1], [0,1]}
function values = get_posiblesValues(n)

	values = {}
	for i = 1:n
		values(i) = 0:1; 
	end

end

% sets: set of vectors
function result = cartesian_product(sets)
    c = cell(1, numel(sets));
    [c{:}] = ndgrid( sets{:} );
    result = cell2mat( cellfun(@(v)v(:), c, 'UniformOutput',false) );
end
