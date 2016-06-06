function run()
	rand('seed', 10);
	test_network.with_terrain('../../doc/data/terrain8.txt', 1, [10 8 6 1], 2, 0.0001, 0.1, 0.5, 0.9, 0.1, 0.1, 2, true);
end
