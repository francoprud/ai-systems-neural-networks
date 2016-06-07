function networkWeights = run()
	rand('seed', 0.55);
	networkWeights = test_network.with_terrain('../../doc/data/terrain8.txt', 1, [6 5 4 1], 1, 0.0005, 0.05, 1.25, 0, 0, 0, 0, true);
end
