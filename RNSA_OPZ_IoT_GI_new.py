# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 09:03:01 2018

@author: User
"""
import math
import random
import csv
#import numpy as np
from memory_profiler import profile
#from memory_profiler import memory_usage

#from sklearn.model_selection import train_test_split

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

def generate_random_antibody(benign_traffic_train, parameters):
    #format: [[center], radius]
    radius = parameters["radius"]
    center = []
    for i in range(1,len(benign_traffic_train[0])):
        center.append(random.uniform(0,1))
    return [center, radius]


precision = 10
fp = open('train_memory_iot_gini.log', 'w+')
@profile(precision=precision, stream=fp)

def generate_detectors(benign_traffic_train, population_size, parameters, self_class, non_self_class):
    antibodies = []
    original_self_class = [x for x in benign_traffic_train if x[0] == self_class]
    while len(antibodies) < population_size:
        self_class = original_self_class #this allows the selection above to happen only once
        proposed_antibody = generate_random_antibody(benign_traffic_train,parameters)
        
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
        if euclidean_distance(a[0], x[1:])* (1.0/n) < a[1]:
            return non_self_class
    return self_class


#function for calculating accuracy

precision = 10
fp = open('train_memory_iot_gini.log', 'w+')
@profile(precision=precision, stream=fp)
  
def get_accuracy(detectors,normalized_benign_scan_test, self_class, non_self_class):
    correct = 0.0
    incorrect = 0.0
    
    for x in normalized_benign_scan_test:
        acc = predict(detectors, x, self_class, non_self_class )
        if x[0] == acc:
            correct += 1
            #print("correct")
        else:
            incorrect += 1
            #print("incorrect")
    accuracy = float(correct) / float(len(normalized_benign_scan_test))
    
    return accuracy


train_data = getdata('Reduce_Train_Data_GI.csv')

test_data = getdata('Reduce_Test_Data_GI.csv')

min_maxtr = dataset_minmax(train_data)

min_maxts = dataset_minmax(test_data)

norm_train = normalize_dataset(train_data, min_maxtr)


norm_test = normalize_dataset(test_data, min_maxts)


parameters = {}
parameters["radius"] = 0.6
det = generate_detectors(norm_train, 125786, parameters, '1', '0')
acc = get_accuracy(det, norm_test, '1', '0')
print(acc)

#mem_acc = memory_usage(acc)

#def get_resul():
    #acc_thres = []
    #for i in np.arange(0, 1.0, 0.1):
        #parameters = {}
        #parameters["radius"] = i
        #det = generate_detectors(norm_train, 125786, parameters, '1', '0')
        #acc = get_accuracy(det, norm_test, '1', '0')
        #acc_thres.append(i)
        #acc_thres.append(acc)
    #return acc_thres

#Value = get_resul()

#np.savetxt('Accuracy', Value, delimiter=', ')
    


#det = generate_detectors(benign_traffic_train, 125786, parameters, '1', '0')
#acc = get_accuracy(det, benign_traffic_test, '1', '0' )

#with open('complete_normalized_training_set.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #for x in benign_traffic_train:
        #writer.writerow(x)

#with open('complete_normalized_testing_set.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #for x in benign_traffic_test:
        #writer.writerow(x)


#with open('accuracy_threshold.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #for i in numpy.arange(0.65, 0.75, 0.01):
        #acc_thre = []
        #parameters["radius"] = i
        #det = generate_detectors(norm_train, 82333, parameters, '0', '1')
        #acc = get_accuracy(det, norm_test, '0', '1' )
        #acc_thre.append(acc)
        #acc_thre.append(i)
        #writer.writerow(acc_thre)


        
#with open('labels.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #for x in benign_traffic_test:
        #pred = predict(det, x, '1', '0')
        #x.append(pred)
        #writer.writerow(x)

#with open('PCA_Training_data_transform.csv', newline = '') as f:
    #Data = []
    #reader = csv.reader(f)
    #for row in reader:
        #Data.append(row)
        
#with open('Detectors.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #for x in det:
        #writer.writerow(x)
        

        
#with open('b_distance.csv', 'w', newline ='') as f:
    #writer = csv.writer(f)
    #for x in benign_traffic_test:
        #for a in det:
            #j = len(a[1])
            #distance = euclidean_distance(a[0], x[1:]) * (1.0 / j)
        #x.append(distance)
        #writer.writerow(x)






        




