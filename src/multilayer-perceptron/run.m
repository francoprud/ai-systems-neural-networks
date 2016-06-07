function run()
	rand('seed', 10);
	test_network.with_terrain('../../doc/data/terrain2.txt', 1, [20 15 10 1], 2, 0.0001, 0.1, 0.75, 0.9, 0.025, 0.05, 2, true);
end
