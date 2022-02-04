from deap import algorithms, base, creator, tools
import random
import numpy as np
import time
import multiprocessing
from helpersGeneticAlgo import evalTSP, orderedCrossover, twoOptMutation, eaSimple

#########################
### GENETIC ALGORITHM ###
#########################
def geneticModel(timeLimit, distanceMatrix, individualSize, populationSize, crossoverPB, mutationPB, nrGenerations, notImprovingLimit, keepHistory=False):
    toolbox = base.Toolbox()
    INDIVIDUAL_SIZE = individualSize

    # CREATE BASE TYPES
    # Fitness (=path length) has negative weight because it will be minimized
    # (a minimizing fitness is built using negatives weights, while a maximizing fitness has positive weights)
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    # create individual class
    # Individual is identified by a list of floats, every float indicates the id of a node (a gene)
    creator.create('Individual', list, fitness=creator.FitnessMin)
    # indices indicates the list of individuals that compose a population.
    # Composed by a random sample taken from the range of len equal to the size of the population we want.
    # Random sample avoids the creation of duplicates in a single individual (each hole/gene is visited once)
    toolbox.register('indices', random.sample, range(INDIVIDUAL_SIZE), INDIVIDUAL_SIZE)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # SETUP GENETIC STEPS
    # The following steps are performed in the order: mate, mutate, select
    #toolbox.register('mate', tools.cxPartialyMatched)
    toolbox.register('mate', orderedCrossover)
    #toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register('mutate', twoOptMutation)
    # Tournsize indicates the nr of random individuals to take at each generation to extract the best fit.
    # Taking now 5% of the population, among this subset, the best is taken.
    # Tournament selects 5% of the population at random and keeps the fittest individual, this is cycled until
    # a number of individuals equal to populationSize is extracted from the original population, these are
    # the offsprings of the next generation
    toolbox.register('select', tools.selTournament, tournsize=int(round(populationSize*0.05)))
    toolbox.register('evaluate', evalTSP, distanceMatrix=distanceMatrix)

    # LAUNCH OPTIMIZATION
    history = tools.History()
    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    pop = toolbox.population(n=populationSize)
    history.update(pop)

    # Hall of fame will store only one best individual of each generation
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    sta = time.time()
    #pop, logb = algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 30, stats=stats, halloffame=hof)
    pop, logb, generationLog = eaSimple(pop, toolbox, crossoverPB, mutationPB, nrGenerations, stats=stats, halloffame=hof,
                                    keepHistory=keepHistory, timeLimit=timeLimit, notImprovingLimit=notImprovingLimit, verbose=False)
    return pop, logb, hof, generationLog