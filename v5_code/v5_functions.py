import numpy as np
import math
import random
from collections import Counter
import pandas as pd
import glob

import time

'filer.py'
'----------------------------------------------------------------------------------------------------------'


def posPool(path, vectorSize):
    # path = './Sars-Cov-2 Project/v5_Dataset/'
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
            maxPool_windowSize = int(vectorSize / 210) + 1
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
    # path = './Sars-Cov-2 Project/v5_Dataset/'
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


def getFeature(path, file):
    # path = './Sars-Cov-2 Project/v5_Dataset/'
    files = glob.glob(path + 'featsVector/*.csv')

    sequences = pd.read_csv(file, header=None).values.ravel()  # get the sequence
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

        pd.DataFrame(featureList).to_csv(path + 'feature/features_' + str(filterIndex) + '.csv', header=None,
                                         index=None)


'----------------------------------------------------------------------------------------------------------'

'appearance.py'
'----------------------------------------------------------------------------------------------------------'


def sameFeature(path):
    # path = './Sars-Cov-2 Project/v5_Dataset/'

    feature_files = glob.glob(path + 'feature/*')

    p_value = 0
    for file in feature_files:
        if p_value == 0:
            feature_data = list(pd.read_csv(file, header=None).values.ravel())
            p_value = 1
        elif p_value == 1:
            feature_data_new = list(pd.read_csv(file, header=None).values.ravel())
            for var in feature_data_new:
                feature_data.append(var)

    CountFeature_dict = Counter(feature_data)

    RepeatList = []
    nonRepeatList = list(set(feature_data))
    for key in CountFeature_dict.keys():
        value = CountFeature_dict.get(key)
        if value != 1:
            RepeatList.append(key)

    pd.DataFrame(RepeatList).to_csv(path + 'Repeat_feature_List.csv', header=None, index=None)
    pd.DataFrame(nonRepeatList).to_csv(path + 'nonRepeat_feature_List.csv', header=None, index=None)


def CG_content_chect(path, min_CG=0.35, max_CG=0.65):
    # path = './Sars-Cov-2 Project/v5_Dataset/'
    p_value = 0

    feature_file = path + 'nonRepeat_feature_List.csv'
    features = pd.read_csv(feature_file, header=None).values.ravel()

    new_features = []
    for feature in features:
        if 'C' in feature and 'G' in feature:
            CountFeature_dict = Counter(feature)
            CG_content = CountFeature_dict.get('C') + CountFeature_dict.get('G')

            if len(feature) * min_CG < CG_content < len(feature) * max_CG:
                new_features.append(feature)
            else:
                pass
        else:
            pass

    Feature_List = list(set(new_features))
    if len(Feature_List) == 0:
        p_value = 2
    else:
        p_value = 1
    pd.DataFrame(Feature_List).to_csv(path + 'nonRepeat_feature_List.csv', header=None, index=None)

    return p_value


def get_appearance(feature_file, seq_file, high_type=1, accuracy=0.95):
    '''
    :msg: get the accuracy of a feature sequence higher than 95% or lower than 5%
    :param feature_file: --- {str} --- the path of the feature sequence file
    :param seq_file: --- {str} --- the path of the sequence file
    :param high_type: --- {int} --- a compare param (defaults to 1)
    :param accuracy: --- {float} --- the limitation of the compare (defaults to 0.95)
    :return: the DataFrame with the appearance of each feature sequence
    '''
    features = pd.read_csv(feature_file, header=None).values.ravel()  # get 3827 features from the CNN model

    # features = pd.read_csv(feature_file, header=None)
    # features = features[1:][1].values.ravel()

    # seq = pd.read_csv(seq_file, header=None).values.ravel()
    seq = pd.read_csv(seq_file, header=None).values.ravel()

    np.random.shuffle(seq)
    seq = pd.DataFrame(seq[:5000])

    total_num = len(seq)
    featureDic = {}
    count = 1
    for feature in features:
        print('feature ---- No. {}      with {} Total'.format(count, len(features)))
        count += 1

        count_feature = seq[0].apply(lambda x: x.count(feature))
        available_count = len(count_feature[count_feature == 1])

        if high_type == 1 and available_count / total_num >= accuracy:
            featureDic[feature] = available_count / total_num
        elif high_type != 1 and available_count / total_num <= 1 - accuracy:
            featureDic[feature] = available_count / total_num

    featureDF = pd.Series(featureDic).to_frame()
    return featureDF


def calculateAppearance(path, file):
    # path = './Sars-Cov-2 Project/v5_Dataset/'

    p_value = 0

    order = ['delta', 'beta', 'alpha', 'gamma']  # original order = ['alpha', 'beta', 'gamma', 'delta']
    for save_fileName in order:
        feature_file = path + 'nonRepeat_feature_List.csv'
        seq_file = path + 'Seq_and_SeqName/seq_GISAID_' + save_fileName + '.csv'

        print('Get the appearance in  -- {} --  virus'.format(save_fileName))
        if save_fileName == 'delta':
            seq_file = file
            if p_value == 0:
                featureDF = get_appearance(feature_file, seq_file, high_type=1, accuracy=0.95)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                if len(temp_feature) == 0:
                    p_value = 2
                else:
                    p_value = 1
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)

            elif p_value == 1:
                feature_file = path + 'result/temp_feature.csv'
                featureDF = get_appearance(feature_file, seq_file, high_type=1, accuracy=0.95)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                if len(temp_feature) == 0:
                    p_value = 2
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)

            elif p_value == 2:
                pass

        else:
            if p_value == 0:
                featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.95)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                if len(temp_feature) == 0:
                    p_value = 2
                else:
                    p_value = 1
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)

            elif p_value == 1:
                feature_file = path + 'result/temp_feature.csv'
                featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.95)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                if len(temp_feature) == 0:
                    p_value = 2
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)

            elif p_value == 2:
                pass

    return p_value


def commonFeatureAppearance(path, forward_primer):
    # path = './Sars-Cov-2 Project/v5_Dataset/'

    files = glob.glob(path + 'Seq_appearance/*')

    Name_list = []
    p_value = 0
    for file in files:
        file_name = file.split('/')[-1].split('.')[0]
        Name_list.append(file_name)

        if p_value == 0:
            data = pd.read_csv(file, header=None)
            data = data[0][data[0].notna()].values.ravel()
            p_value = 1
        elif p_value == 1:
            data_new = pd.read_csv(file, header=None)
            data_new = data_new[0][data_new[0].notna()].values.ravel()

            data = [x for x in data if x in data_new]

    Appearance_list = []
    for seq in data:
        print('\nFor the sequence: {}\n'.format(seq))
        temp_list = []
        for file in files:
            file_name = file.split('/')[-1].split('.')[0]
            data_appearance = pd.read_csv(file, header=None)
            appearance = data_appearance[data_appearance[0] == seq][1].values.ravel()[0]
            # print('{} :  {}'.format(file_name, appearance))
            temp_list.append(appearance)
        Appearance_list.append(temp_list)

    Appearance_DF = pd.DataFrame(Appearance_list)
    Appearance_DF.index = data
    Appearance_DF.columns = Name_list

    Appearance_DF = Appearance_DF[['feature_alpha', 'feature_beta', 'feature_gamma', 'feature_delta']]

    Appearance_DF = Appearance_DF.sort_values(by=['feature_delta'], ascending=[False])
    Appearance_DF.to_csv(path + 'result/' + forward_primer + '_Appearance_DataFrame.csv')


'----------------------------------------------------------------------------------------------------------'
