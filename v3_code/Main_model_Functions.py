'''
The functions that used in the CNN model
'''

import numpy as np
import pandas as pd
from main_Functions import *
import glob
import tensorflow.compat.v1 as tf
import random


def oneHot(array, size):
    output = []
    for i in range(len(array)):
        temp = np.zeros(size)
        temp[int(array[i])] = 1
        output.append(temp)
    return np.array(output)


# function to declare easily the weights only by shape
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# function to declare easily the bias only by shape
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def getBatch_run(data, labels, size, run, vector, sampleSize):
    infLimit = run * size
    supLimit = infLimit + size
    if supLimit > len(data):
        supLimit = len(data)
    batch = []
    for i in range(infLimit, supLimit):
        batch.append(vector[i])
    outData = []
    outLabels = []
    for i in range(len(batch)):
        sample = np.zeros(sampleSize)
        for j in range(0, len(data[batch[i]])):
            if data[batch[i]][j] == 'C':
                sample[j] = 0.25
            elif data[batch[i]][j] == 'T':
                sample[j] = 0.50
            elif data[batch[i]][j] == 'G':
                sample[j] = 0.75
            elif data[batch[i]][j] == 'A':
                sample[j] = 1.0
            else:
                sample[j] = 0.0
        outData.append(sample)
        outLabels.append(labels[batch[i]])
    return np.array(outData), np.array(outLabels)


def getBatch(data, labels, size, sampleSize):
    index = []
    for i in range(len(data)):
        index.append(i)
    batch = random.sample(index, size)
    outData = []
    outLabels = []
    for i in range(len(batch)):
        sample = np.zeros(sampleSize)
        for j in range(0, len(data[batch[i]])):
            if (data[batch[i]][j] == 'C'):
                sample[j] = 0.25
            elif (data[batch[i]][j] == 'T'):
                sample[j] = 0.50
            elif (data[batch[i]][j] == 'G'):
                sample[j] = 0.75
            elif (data[batch[i]][j] == 'A'):
                sample[j] = 1.0
            else:
                sample[j] = 0.0
        outData.append(sample)
        outLabels.append(labels[batch[i]])
    return np.array(outData), np.array(outLabels)
