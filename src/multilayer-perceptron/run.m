function run()
	rand('seed', 10);
	test_network.with_terrain('../../doc/data/terrain2.txt', 1, [20 1], 1, 0.0001, 0.05, 0.75, 0, 0, 0, 0, true);
end
