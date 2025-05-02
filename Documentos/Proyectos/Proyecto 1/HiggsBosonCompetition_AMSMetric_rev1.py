# -*- coding: utf-8 -*-
"""
Evaluation metric for the Higgs Boson Kaggle Competition,
as described on:
https://www.kaggle.com/c/higgs-boson/details/evaluation

@author: Joyce Noah-Vanhoukce
Created: Thu Apr 24 2014

"""

import os
import csv
import math
import pandas as pd
import pickle


def create_solution_dictionary(solution):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """

    # Este proceso toma tiempo: una vez generado el diccionario se salva a disco.
    # if os.path.isfile("{}_dict.pkl".format(solution)):
    #     with open("{}_dict.pkl".format(solution), 'rb') as f:
    #         solnDict = pickle.load(f)
    #         return (solnDict)


    solnDict = {}
    df = pd.read_csv(solution)

    for i, row in df.iterrows():
        solnDict[row.EventId] = (row.Label, row.Weight)

    f = open("{}_dict.pkl".format(solution),"wb")
    pickle.dump(solnDict,f)
    f.close()

    return solnDict

        
def check_submission(submission, Nelements):
    """ Check that submission RankOrder column is correct:
        1. All numbers are in [1,NTestSet]
        2. All numbers are unqiue
    """
    df = pd.read_csv(submission)

    if len(df['EventId'].unique()) != Nelements:
        print('RankOrder column must contain unique values')
        exit()
    else:
        return True

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)


def AMS_metric(solution, submission):
    """  Prints the AMS metric value to screen.
    Solution File header: EventId, Class, Weight
    Submission File header: EventId, Class
    """

    # solutionDict: key=eventId, value=(label, class)
    solutionDict = create_solution_dictionary(solution)

    numEvents = len(solutionDict)

    signal = 0.0
    background = 0.0
    if check_submission(submission, numEvents):
        df = pd.read_csv(submission)
        #with open(submission, 'rb') as f:
        #    sub = csv.reader(f)
        #    sub.next() # header row
        for i, row in df.iterrows():
            if row[1] == 1: # only events predicted to be signal are scored
                if solutionDict[row[0]][0] == 1:
                    signal += float(solutionDict[row[0]][1])
                elif solutionDict[row[0]][0] == 0:
                    background += float(solutionDict[row[0]][1])
     
        print('signal = {0}, background = {1}'.format(signal, background))
        ams = AMS(signal, background)
        print('AMS = ' + str(ams))

        return(ams)



if __name__ == "__main__":

    # Los datos se obtienen habiendo corrido primero el proyecto1_data_preparation.py
    path = "/Users/javierpreciozzi/Documents/facultad/taa/taa_2021/Proyecto_1/prueba_evaluacion"
    solutionFile = "{}/ground_truth_with_weights.csv".format(path)
    submissionFile = "{}/random_submission.csv".format(path)
    
    AMS_metric(solutionFile, submissionFile)

    AMS_metric(solutionFile, solutionFile)
