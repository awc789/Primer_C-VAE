import numpy as np
import math
import random
import pandas as pd
import glob
import time

def posPool(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    files = glob.glob(path + 'filter/*.csv')
    for file in files:
        filterIndex = file.split('/')[-1].split('.')[0].split('_')[-1]
        data = pd.read_csv(file, header=None).values
        numberWindows = 210

        sizeData = np.shape(data)

        print(sizeData)

        maxPool = np.zeros(shape=(sizeData[0], numberWindows))
        posPool = np.zeros(shape=(sizeData[0], numberWindows))

        for i in range(0, sizeData[0]):
            maxPool_windowSize = 148
            pad_left_HPool = 0
            max = -1e6
            index = pad_left_HPool
            position = -1
            indexMax = 0
            for j in range(0, sizeData[1]):
                if data[i][j] > max:
                    max = data[i][j]
                    position = j
                index = index + 1
                if index == maxPool_windowSize or j == sizeData[1] - 1:
                    maxPool[i][indexMax] = max
                    posPool[i][indexMax] = position
                    max = -1e6
                    position = -1
                    index = 0
                    indexMax = indexMax + 1

        pd.DataFrame(maxPool).to_csv(path + 'maxPool/maxPool_' + str(filterIndex) + '.csv', header=None, index=None)
        pd.DataFrame(posPool).to_csv(path + 'posPool/posPool_' + str(filterIndex) + '.csv', header=None, index=None)


def creatFeatVector(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    pos_path = path + 'posPool/'
    path_seq = path + 'filter_seq/'
    files = glob.glob(pos_path + '/*.csv')

    # Parameters
    vectorSize = 31079
    numberFilters = 21
    padding = 10

    for file in files:
        filterIndex = file.split('/')[-1].split('.')[0].split('_')[-1]
        print('Processing...   Loop 1 -- Index {}'.format(filterIndex))

        posMatrix = pd.read_csv(file, header=None).values
        matrix = pd.read_csv(path_seq + 'filter_seq.csv', header=None).values.ravel()

        outData = []
        for i in range(len(matrix)):
            sample = np.zeros(vectorSize)
            for j in range(len(matrix[i])):
                if matrix[i][j] == 'C':
                    sample[j] = 0.25
                elif matrix[i][j] == 'T':
                    sample[j] = 0.50
                elif matrix[i][j] == 'G':
                    sample[j] = 0.75
                elif matrix[i][j] == 'A':
                    sample[j] = 1.0
                else:
                    sample[j] = 0.0
            outData.append(sample)

        matrix = np.array(outData)
        sizePosMatrix = np.shape(posMatrix)
        dataDNA = [[0 for i in range(210 * numberFilters)] for j in range(sizePosMatrix[0])]
        sizeDNAMatrix = np.shape(matrix)

        for i in range(sizePosMatrix[0]):
            temp = ((matrix[i]))
            for j in range(sizePosMatrix[1]):
                coef = int(posMatrix[i][j])
                for k in range(padding + 1):
                    if (coef + k) < len(temp):
                        dataDNA[i][j * numberFilters + padding + k] = temp[coef + k]
                    if (coef - k) >= 0 and (coef - k) < len(temp):
                        dataDNA[i][j * numberFilters + padding - k] = temp[coef - k]

        dataDNAString = [[0 for i in range(210 * numberFilters)] for j in range(sizePosMatrix[0])]

        for i in range(sizePosMatrix[0]):
            for j in range(210 * numberFilters):
                if dataDNA[i][j] == 0.25:
                    dataDNAString[i][j] = "C"
                elif dataDNA[i][j] == 0.50:
                    dataDNAString[i][j] = "T"
                elif dataDNA[i][j] == 0.75:
                    dataDNAString[i][j] = "G"
                elif dataDNA[i][j] == 1.00:
                    dataDNAString[i][j] = "A"
                else:
                    dataDNAString[i][j] = "N"

        dataDNAFeatures = [[0 for i in range(210)] for j in range(sizePosMatrix[0])]
        for i in range(sizePosMatrix[0]):
            for j in range(210):
                dataDNAFeatures[i][j] = str("")

        for i in range(sizePosMatrix[0]):
            indexFeature = 0
            feature = 0
            for j in range(210 * numberFilters):
                dataDNAFeatures[i][feature] = str(dataDNAFeatures[i][feature]) + str(dataDNAString[i][j])
                indexFeature = indexFeature + 1
                if indexFeature == numberFilters:
                    feature = feature + 1
                    indexFeature = 0

        featsVector = []
        for i in range(sizePosMatrix[0]):
            for j in range(210):
                count = featsVector.count(dataDNAFeatures[i][j])
                if count == 0:
                    if "N" not in dataDNAFeatures[i][j]:
                        featsVector.append(dataDNAFeatures[i][j])

        pd.DataFrame(featsVector).to_csv(path + 'featsVector/featsVector_' + str(filterIndex) + '.csv',
                                         header=None, index=None)
        pd.DataFrame(dataDNAFeatures).to_csv(path + 'dataDNAFeatures/dataDNAFeatures_' + str(filterIndex) + '.csv',
                                             header=None, index=None)


def getFeature(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    files = glob.glob(path + 'featsVector/*.csv')

    variant = ['alpha', 'beta', 'gamma', 'delta']
    seq_file = path + 'Seq_and_SeqName/seq_GISAID_' + variant[3] + '.csv'
    sequences = pd.read_csv(seq_file, header=None).values.ravel()  # get the sequence
    np.random.shuffle(sequences)
    sequences = sequences[:1000]

    for file in files:
        filterIndex = file.split('/')[-1].split('.')[0].split('_')[-1]
        print('Processing...   Loop 1 -- Index {}'.format(filterIndex))

        vector = pd.read_csv(file, header=None).values.ravel()
        print('Sequence Size: {}'.format(len(sequences)))
        print('featVector Size: {}'.format(len(vector)))

        featureList = []
        count = 0
        for seq in sequences:
            count += 1
            if count % 100 == 0:
                print('Calculating the features in sequence No.{}      with {} Total'.format(count, len(sequences)))
            for feature in vector:
                if feature in seq:
                    feature_count = seq.count(feature)
                    if feature_count == 1 and feature not in featureList:
                        featureList.append(feature)

        pd.DataFrame(featureList).to_csv(path + 'feature/features_' + str(filterIndex) + '.csv', header=None, index=None)


if __name__ == '__main__':
    start = time.time()
    path = './Sars-Cov-2 Project/v3_Dataset/'

    posPool(path)
    creatFeatVector(path)
    getFeature(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print(' ')
    print(' ')
    print(' ')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))
