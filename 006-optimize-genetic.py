from os import walk, listdir
import json
from config import TIME_LIMITS, GENETIC_ALGO_LOOPS
import sys
from deap import algorithms, base, creator, tools
import random
import numpy as np
import time
import multiprocessing
from helpersGeneticAlgo import evalTSP, orderedCrossover, twoOptMutation, eaSimple
from genetic_model import geneticModel

results = {}

with open('results.json') as json_data:
    res = json.load(json_data)

# look for all the folders with points distributions
for (dirpath, dirnames, filenames) in walk('./points/'):
    points = [d for d in dirnames if not d.startswith('.ipynb')]
    print('Found the following folders:', points)
    break

#points = [10]
# check how many folders were found
optimize = input('The following folders where found: {}. Continue with optimization? [y/n]'.format(points))
if str(optimize) == 'n':
    sys.exit('Optimization interrupted by used')

# optimal model configuration found via 004-hypspace-exploration.py
modelConfig = {
    'populationSize': 230,
    'crossoverPB': 0.64,
    'mutationPB': 0.33,
    'nrGenerations': 308,
    'notImprovingLimit': 16
}

# loop through points (folders) and run the optimization
for p in points:
    p = int(p)
    folder = './points/{}/'.format(p)
    # each folder has 10 variations of point distributions
    # get files with points distribution format ready for OPL
    files = [file for file in listdir(folder) if file.endswith('.dat')]
    # loop through each point distribution variation
    for i in range(1,int(len(files))+1):
        # load distance matrix for current file
        npzfile = np.load('./points/{}/{}.npz'.format(p,i))
        distanceMatrix = npzfile['dist']
        # loop through different time limits
        for timeLimit in TIME_LIMITS:
            print('Running optimization for points/{}/{}.dat with time limit {}s'.format(p, i, timeLimit))
            # update model configuration
            modelConfig['distanceMatrix'] = distanceMatrix
            modelConfig['individualSize'] = p
            modelConfig['timeLimit'] = timeLimit
            # loop genetic algorithm more then once (to cope with possible randomness of results)
            runStats = {
                'runs': 0,
                'runningTimes': [],
                'objFunValues': [],
                'solutions': []
            }
            startTime = time.time()
            # the genetic algorithm is run more than once, to mitigate randomness of results
            for loop in range(GENETIC_ALGO_LOOPS):
                print('  Loop #{}'.format(loop+1))

                pop, logb, hof, generationLog = geneticModel(**modelConfig)

                # check if any solution was found (otherwise no fit was computed)
                if generationLog == False:
                    break
                # update loop stats
                runStats['runs'] += 1
                runStats['objFunValues'].append(hof.keys[0].values[0])
                runStats['solutions'].append(hof.items[0])
                runStats['runningTimes'].append(time.time()-startTime)
                # check time limit constraint
                if time.time()-startTime >= timeLimit:
                    break

            if generationLog == False:
                continue
            # after the loop, keep only best overall solution and total time
            results.setdefault(p, {})
            results[p].setdefault(i, {})
            # computing both the min and the mean value. In a real application we could keep the minimum value found
            # but for evaluation purposes lets keep the mean value of the 3 runs of the algorithm
            minObjFunValue = round(min(runStats['objFunValues']),13) # round to 13 for comparability with OPL
            meanObjFunValue = round(np.mean(runStats['objFunValues']),13) # round to 13 for comparability with OPL

            # retrieve optimal solution found with OPL
            optimalFunValue = res[str(p)][str(i)]['9999']['objFunValue']
            deltaFromOpt = ((meanObjFunValue/optimalFunValue)-1)*100

            results[p][i][timeLimit] = {
                'runningTime[ms]': np.sum(runStats['runningTimes'])*1000,
                'deltaFromOpt[%]': deltaFromOpt,
                'objFunValue': meanObjFunValue,
                'runs': runStats['runs'],
                'meanObjFunValue': meanObjFunValue
            }
            # store solution (associated to the min value)
            bestSolutionIndex = runStats['objFunValues'].index(min(runStats['objFunValues']))
            np.savez('./points/{}/{}_{}_gen_sol'.format(p,i,timeLimit), sol=runStats['solutions'][bestSolutionIndex])

# store dictonary with unsolved operations to file for later analysis
with open("results_genetic.json", "wt") as fout:
    json.dump(results, fout)

print('Optimization process finished')