import time
import numpy as np
import random
from deap import tools

# Some of the following functions were taken from:
# https://github.com/DEAP/deap/blob/master/deap/algorithms.py

############################
### EVALUATION FUNCTIONS ###
############################
# Returns the fitness that we want to minimize
# (the length of a route, as sum of distances between the nodes)
def evalTSP(individual, distanceMatrix):
    # Initialize distance with distance from last individual to the first one.
    # This is the distance needed to get back from the last node to the first one, to close the cycle
    distance = distanceMatrix[individual[-1]][individual[0]]
    # Calculate the distance between the various points, so if we have [1,2,3,4,5]
    # the first cycle will give gene1=1, gene2=2
    # the second cycle will give gene1=2, gene2=3...
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distanceMatrix[gene1][gene2]
    return distance,

###################################
### CROSSOVER(MATING) FUNCTIONS ###
###################################
# Custom functions to do order crossover
def crossIndividuals(ind1, ind2, geneIds):
    geneMin, geneMax = sorted(geneIds)
    start = ind1[:geneMin]
    end = ind1[geneMax:]
    middle = [gene for gene in ind2 if gene not in start+end]
    return start+middle+end

def orderedCrossover(ind1, ind2):
    # print('### 3. CROSSOVER')
    # Choose 2 random genes and use those as cut points
    geneIds = np.random.choice(range(len(ind1)), 2, replace=False)
    ind1[:] = crossIndividuals(ind1, ind2, geneIds)
    ind2[:] = crossIndividuals(ind2, ind1, geneIds)
    return ind1, ind2

############################
### MUTATION FUNCTIONS ###
############################
# Custom function for 2-opt mutation
def twoOptMutation(ind):
    # print('### 4. MUTATION')
    geneMin, geneMax = sorted(np.random.choice(range(len(ind)), 2, replace=False))
    start = ind[:geneMin]
    end = ind[geneMax:]
    middle = list(reversed(ind[geneMin:geneMax]))
    ind[:] = start+middle+end
    return ind,


###########################
### COMPLETE ALGORITHM ###
###########################
def outOfTime(startTime, timeLimit):
    if timeLimit == 9999:
        return False
    if time.time()-startTime > timeLimit:
        return True
# Function to implement early stopping
def notImproving(fitnessHistory, limit):
    if limit == 0:
        return False
    if len(fitnessHistory)>=limit:
        bestPrevFit = min(fitnessHistory[:-1])
        # Terminate if there is no improvement over 0.0001 during the last "limit" generations
        # fit values compared to the best ever found in all previous generations
        if min(fitnessHistory[-limit:]) - bestPrevFit > 1e-4:
            print('   Early stopping, no improvement for {} iterations'.format(limit))
            return True  

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring. Offspring has the same size as population
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            # delete fitness value which will be updated later
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring
    
def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, keepHistory=False, timeLimit=9999, notImprovingLimit=0, verbose=__debug__):
    
    startTime = time.time()
    generationLog = []
    avgFitnessHistory = []
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Check time limit constraint
    if outOfTime(startTime, timeLimit):
        return population, logbook, False

    # Evaluate the individuals with an invalid fitness.
    # Invalid means that the fitness has not yet been computed
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    if outOfTime(startTime, timeLimit):
        return population, logbook, False
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    # Populate hall of fame with best individuals
    if halloffame is not None:
        halloffame.update(population)

    # Check early stopping constraint and time limit constraint
    if notImprovingLimit>0:
        avgFitnessHistory = [np.mean([ind.fitness.values[0] for ind in population])]
    if outOfTime(startTime, timeLimit) or notImproving(avgFitnessHistory, notImprovingLimit):
        return population, logbook, generationLog

    # update statistics
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # Keep history is needed only for the 007-genetic-algo-animation file, to reproduce the
    # full evolution of the algorithm
    if keepHistory:
        generationLog = [[toolbox.clone(ind) for ind in population]]

    if outOfTime(startTime, timeLimit):
        return population, logbook, generationLog
        
    # Begin the generational process (this is run for every generation)
    for gen in range(1, ngen + 1):
        # print('### 1. SELECTING NEXT GENERATION')
        # Select the next generation candidates individuals.
        # The next command runs the selTournament function (tools.selTournament on genetic_model.py) which selects
        # batches of individuals and keeps the best one, until a number equal to len(population) is extracted.
        # These will be the new generation candidates that will go through crossover and mutation.
        offspring = toolbox.select(population, len(population))
        
        # print('### 2. APPLY CROSSOVER AND MUTATION')
        # Vary the pool of individuals (apply crossover and mutation, with given probabilities)
        # the following command will call orderedCrossover first, and then twoOptMutation, as defined in genetic_model.py
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        if outOfTime(startTime, timeLimit):
            return population, logbook, generationLog
        
        # Evaluate the individuals with an invalid fitness
        # (update fitness of new offsprings)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        if keepHistory:
            generationLog.append(offspring)
        if notImprovingLimit>0:
            avgFitnessHistory.append(np.mean([ind.fitness.values[0] for ind in population]))
        if outOfTime(startTime, timeLimit) or notImproving(avgFitnessHistory, notImprovingLimit):
            return population, logbook, generationLog
        
    return population, logbook, generationLog