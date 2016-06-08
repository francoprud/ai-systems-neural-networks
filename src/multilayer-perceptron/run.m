function networkWeights = run()
	rand('seed', 0.55);
	networkWeights = test_network.with_terrain('../../doc/data/terrain8.txt', 1, [6 5 4 1], 1, 0.0007, 0.05, 0.75, 0.9, 0.005, 0.1, 1, true);
end
