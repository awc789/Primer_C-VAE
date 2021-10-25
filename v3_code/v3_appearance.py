import numpy as np
import pandas as pd
import glob
from collections import Counter
from main_Functions import *
import time


def sameFeature(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'

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

    pd.DataFrame(RepeatList).to_csv(path + 'Repeat_feature_List.csv',header=None, index=None)
    pd.DataFrame(nonRepeatList).to_csv(path + 'nonRepeat_feature_List.csv',header=None, index=None)


def get_appearance(feature_file, seq_file, high_type = 1, accuracy = 0.95):
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


def calculateAppearance(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'

    files = glob.glob(path + 'Variant_virus/*')
    p_value = 0

    order = [3, 1, 0, 2]  # original order = ['alpha', 'beta', 'gamma', 'delta']
    for i in order:
        file = files[i]
        file = file.split('/')[-1]
        save_fileName = file.split('.')[0].replace('GISAID_', '')
        # save_Fasta(path, file, save_fileName)

        # feature_file = path + 'Repeat_feature_List.csv'
        feature_file = path + 'nonRepeat_feature_List.csv'
        seq_file = path + 'Seq_and_SeqName/seq_' + file + '.csv'

        print('Get the appearance in  -- {} --  virus'.format(save_fileName))
        if save_fileName == 'delta':
            if p_value == 0:
                featureDF = get_appearance(feature_file, seq_file, high_type=1, accuracy=0.99)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)
                p_value = 1

            elif p_value != 0:
                feature_file = path + 'result/temp_feature.csv'
                featureDF = get_appearance(feature_file, seq_file, high_type=1, accuracy=0.99)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)

        else:
            if p_value == 0:
                featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.99)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)
                p_value = 1

            elif p_value != 0:
                feature_file = path + 'result/temp_feature.csv'
                featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.99)
                featureDF.to_csv(path + 'Seq_appearance/feature_' + save_fileName + '.csv')

                temp_file = path + 'Seq_appearance/feature_' + save_fileName + '.csv'
                temp_data = pd.read_csv(temp_file)
                temp_feature = temp_data['Unnamed: 0'].values.ravel()
                pd.DataFrame(temp_feature).to_csv(path + 'result/temp_feature.csv', header=None, index=None)


def commonFeatureAppearance(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'

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
    Appearance_DF.to_csv(path + 'result/Appearance_DataFrame.csv')


if __name__ == '__main__':
    start = time.time()
    path = './Sars-Cov-2 Project/v3_Dataset/'

    sameFeature(path)
    calculateAppearance(path)
    commonFeatureAppearance(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print(' ')
    print(' ')
    print(' ')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))


