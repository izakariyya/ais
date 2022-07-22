# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 09:03:01 2018

@author: User
"""
import math
import random
import csv
#import numpy as np
#from memory_profiler import memory_usage

#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import RandomForestClassifier

def getdata(dataset):
    with open(dataset, newline = '') as f:
        rowdata = []
        reader = csv.reader(f)
        for row in reader:
            for i in range(1, len(row)):
                row[i] = float(row[i])
            rowdata.append(row)          
    return rowdata


#function for calculating the eucidean distance between two point
    
def euclidean_distance(x1, x2):
    distance = math.sqrt(sum( (x1 - x2)**2 for x1, x2 in zip(x1, x2)))
    return distance

#Getting the minimum and maximum value for each column
    
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
    
def normalize_dataset(dataset, minmax):
    normdata = []
    for row in dataset:
        for i in range(1, len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        normdata.append(row)          
    return normdata



#normalized_benign_scan_test = getdata('normalized_scan_benign_test.csv')


def generate_random_antibody(norm_train, parameters):
    #format: [[center], radius]
    radius = parameters["radius"]
    center = []
    for i in range(1,len(norm_train[0])):
        center.append(random.uniform(0,1))
    return [center, radius]


def generate_detectors(norm_train, population_size, parameters, self_class, non_self_class):
    antibodies = []
    original_self_class = [x for x in norm_train if x[0] == self_class]
    while len(antibodies) < population_size:
        self_class = original_self_class #this allows the selection above to happen only once
        proposed_antibody = generate_random_antibody(norm_train,parameters)
        
        #select the self class points in each dimension that could be contained in by the proposed antibody
        for i in range(1,len(self_class[0])):
            self_class = [s for s in self_class if s[i] >(proposed_antibody[0][i-1]-proposed_antibody[1]) and s[i] <(proposed_antibody[0][i-1]+proposed_antibody[1])]
#if the self_class list is empty then add the antibody, since there are no points in the self class contained by the hyper-cube containing the hyper-sphere

        if len(self_class) == 0:
            antibodies.append(proposed_antibody)
    
    #check whether the self points selected are actually contained by the hypersphere and not only the hyper cube
    
        else:
            flagged = False
            for s in self_class:
                if euclidean_distance(proposed_antibody[0], s[1:]) < proposed_antibody[1]:
                    flagged = True
            if flagged == False: #if there are no points that are within the hyper-sphere then add the antibody to the population
                antibodies.append(proposed_antibody)
    return antibodies


def predict(antibodies, x, self_class, non_self_class):
    #select the antibodies that could contain the point
    #for every dimension in the antibody center:
    
    for i in range(len(antibodies[0][0])):
        antibodies = [a for a in antibodies if x[i+1] > (a[0][i]-a[1]) and x[i+1] < (a[0][i]+a[1])]
        #further filter the set of antibodies

    for a in antibodies:
        n = len(a[0])
        if euclidean_distance(a[0], x[1:])*(1.0/n) < a[1]:
            return non_self_class
    return self_class


#function for calculating accuracy
    
def get_accuracy(detectors,test_data, self_class, non_self_class):
    correct = 0.0
    incorrect = 0.0
    
    for x in test_data:
        acc = predict(detectors, x, self_class, non_self_class )
        if x[0] == acc:
            correct += 1
            #print("correct")
        else:
            incorrect += 1
            #print("incorrect")
    accuracy = float(correct) / float(len(test_data))
    
    return accuracy



train = getdata('KDDTrainComplete.csv')

test = getdata('KDDTest.csv')

meen_maxee_tr = dataset_minmax(train)

meen_maxee_ts = dataset_minmax(test)

norm_train = normalize_dataset(train, meen_maxee_tr)

norm_test = normalize_dataset(test, meen_maxee_ts)

parameters = {}
parameters["radius"] = 0.5
det = generate_detectors(norm_train, 125972, parameters, '1', '0')
acc = get_accuracy(det, norm_test, '1', '0')
print(acc)

#detmemory =memory_usage(det)

#with open('acc_threshd.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #x = []
    #for i in np.arange(0.6, 1.0, 0.1):
        #parameters = {}
        #parameters["radius"] = i
        #det = generate_detectors(norm_train, 345814, parameters, '1', '0')
        #acc = get_accuracy(det, norm_test, '1', '0')
        #x.append(acc)
        #x.append(i)
        #writer.writerow(x)

