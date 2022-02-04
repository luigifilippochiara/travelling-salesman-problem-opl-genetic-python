import numpy as np

# range of time limits to test the algorithm with
TIME_LIMITS = [0.1,1,10,30,80,9999]

# Genetic algo specific
GENETIC_ALGO_LOOPS = 3
EXPLORATION_SPACE = {
    'populationSize': np.arange(100,501,50),
    'crossoverPB': np.arange(0.4,0.95,0.05),
    'mutationPB': np.arange(0.1,0.65,0.05),
    'nrGenerations': np.arange(50,550,50),
    'notImprovingLimit': np.arange(5,35,5)
}