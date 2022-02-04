import h5py
import numpy as np
import telegram_send
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import json
import time
from genetic_model import geneticModel
from os import walk, listdir
from config import EXPLORATION_SPACE
import sys

# parameter space
space = {
    'populationSize': hp.choice('populationSize', EXPLORATION_SPACE['populationSize']),
    'crossoverPB': hp.choice('crossoverPB', EXPLORATION_SPACE['crossoverPB']),
    'mutationPB': hp.choice('mutationPB', EXPLORATION_SPACE['mutationPB']),
    'nrGenerations': hp.choice('nrGenerations', EXPLORATION_SPACE['nrGenerations']),
    'notImprovingLimit': hp.choice('notImprovingLimit', EXPLORATION_SPACE['notImprovingLimit'])
}

# look for all the folders with points distributions
for (dirpath, dirnames, filenames) in walk('./points/'):
    points = [int(d) for d in dirnames if not d.startswith('.ipynb')]
    break
files = [int(file.replace('.dat','')) for file in listdir('./points/{}/'.format(points[0])) if file.endswith('.dat')]
# load optimal solutions found with OPL
with open('results.json') as json_data:
    res = json.load(json_data)

currentEval = 0
maxEvaluations = 2500
# send telegram notifications every 1/20 progress
notificationSteps = np.round(np.quantile(np.arange(maxEvaluations), np.linspace(0,1,20)))

stepPoints = []
stepDistr = []

# model definition
def create_model(space):
    global currentEval
    global maxEvaluations
    
    # load distance matrix at random
    p = int(np.random.choice(points, size=1)[0])
    i = int(np.random.choice(files, size=1)[0])
    print('Loading file ./points/{}/{}.npz'.format(p,i))
    npzfile = np.load('./points/{}/{}.npz'.format(p,i))
    distanceMatrix = npzfile['dist']

    # load model and optimize
    startTime = time.time()

    modelConfig = {
        'timeLimit': 9999,
        'distanceMatrix': distanceMatrix,
        'individualSize': p,
        'populationSize': space['populationSize'],
        'crossoverPB': space['crossoverPB'],
        'mutationPB': space['mutationPB'],
        'nrGenerations': space['nrGenerations'],
        'notImprovingLimit': space['notImprovingLimit']
    }
    pop, logb, hof, _ = geneticModel(**modelConfig)
    geneticTime = (time.time()-startTime)*1000
    fitness = round(hof.keys[0].values[0], 13) # round to 13 for comparability with OPL
    # load optimal obj function value for current point distribution
    OPLsolution = res[str(p)][str(i)]['9999']
    optimalFunValue = OPLsolution['objFunValue']
    deltaFromOpt = (1-(optimalFunValue/fitness))
    OPLtime = OPLsolution['runningTime[ms]']
    # build custom loss function
    customLoss = 1.2*deltaFromOpt + geneticTime/OPLtime
    #print('custom loss is', customLoss)

    print('Loss of evaluation #{}/{}: {}'.format(currentEval+1, maxEvaluations, customLoss))
    print()
    
    # notify via telegram
    if currentEval in notificationSteps:
        telegram_send.send(['Hyperparameter space exploration reached {} of {} evaluations'.format(currentEval, maxEvaluations)])
    
    stepPoints.append(p)
    stepDistr.append(i)

    currentEval += 1
    return {'loss': customLoss, 'status': STATUS_OK}

# start hyperparameter space exploration
trials = Trials()
best = fmin(create_model, space, algo=tpe.suggest, max_evals=maxEvaluations, trials=trials)

# store results
results = {
    'best': best,
    'trials': trials.trials,
    'results': trials.results,
    'best_trial': trials.best_trial,
    'stepPoints': stepPoints,
    'stepDistr': stepDistr
}
with open('results-space.json', 'w') as fp:
    json.dump(results, fp, default=str)

# send finish notification
telegram_send.send(['Hyperparameter search finished'])