from os import walk, listdir
import re
from subprocess import call, check_call, check_output
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from config import TIME_LIMITS

optimResults = {}

# for detailed comments on the loops, check 001-optimize.py
for (dirpath, dirnames, filenames) in walk('./points/'):
    points = [d for d in dirnames if not d.startswith('.ipynb')]
    break

#points = [10]
for p in points:
    folder = './points/{}/'.format(p)
    files = [file for file in listdir(folder) if file.endswith('.dat')]
    for i in range(int(len(files))):
        distrId = i+1
        for timeLimit in sorted(TIME_LIMITS, reverse=True):
            # read file with raw solution
            try:
                with open('./points/{}/{}_{}_output.txt'.format(p,distrId,timeLimit), "r") as fin:
                    rawOutput = fin.read()
            except FileNotFoundError:
                continue
            
            # RETRIEVE DATA
            # retrieve solving time [mS]
            runningTime = re.search('solving time ~= (.*).', rawOutput)
            runningTime = round(float(runningTime.group(1)),1)
            # retrieve obj function value
            objFunValue = float(re.search(r'with objective (.*)', rawOutput).group(1))
            # retrieve solution type
            isOptimal = False
            solutionType = re.search(r'\/\/ solution \((.*)\)', rawOutput).group(1)
            if solutionType in ['optimal', 'integer optimal, tolerance']:
                isOptimal = True
                optimalFunValue = objFunValue
            # compute distance from optimal solution
            deltaFromOpt = (objFunValue/optimalFunValue-1)*100

            #print(deltaFromOpt, optimalFunValue, objFunValue)

            # store results of current run
            optimResults.setdefault(p,{})
            optimResults[p].setdefault(distrId,{})
            optimResults[p][distrId][timeLimit] = {
                'runningTime[ms]': runningTime,
                'deltaFromOpt[%]': deltaFromOpt,
                'objFunValue': objFunValue
            }

# store dictonary to file for later analysis
with open("results.json", "wt") as fout:
    json.dump(optimResults, fout)

print('Results stored in results.json file')