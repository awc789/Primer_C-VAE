import numpy as np
import pandas as pd
import glob
from main_Functions import *
import time


def get_select_feature(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    main_appearance_file = path + 'result/Appearance_DataFrame.csv'
    appearance_DF = pd.read_csv(main_appearance_file)
    feature = appearance_DF['Unnamed: 0']
    feature.to_csv(path + 'other_virus/main_feature.csv', header=None, index=None)


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
    seq = pd.read_csv(seq_file, header=None)

    total_num = len(seq)
    featureDic = {}
    count = 1
    for feature in features:
        print('feature ---- No. {}'.format(count))
        count += 1

        count_feature = seq[0].apply(lambda x: x.count(feature))
        available_count = len(count_feature[count_feature == 1])

        if high_type == 1 and available_count / total_num >= accuracy:
            featureDic[feature] = available_count / total_num
        elif high_type != 1 and available_count / total_num <= 1 - accuracy:
            featureDic[feature] = available_count / total_num

    featureDF = pd.Series(featureDic).to_frame()
    return featureDF

def calculate_other_virus_Appearance(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    feature_file = path + 'other_virus/main_feature.csv'

    GISAID_SARS_CoV_2 = ['HomoSapiens']
    NCBI_SARS_CoV_2 = ['SARS_CoV_2']
    GISAID_other_virusList = ['Canis', 'Felis_catus', 'Manis_javanica', 'Rhinolophus_affinis', 'Rhinolophus_bat']
    NCBI_other_virusList = ['HAstV_BF34', 'HCoV_229E', 'HCoV_HKU1', 'HCoV_NL63', 'HCoV_OC43', 'MERS_CoV',
                            'SARS_CoV_GDH_BJH01',
                            'SARS_CoV_HKU_39849', 'SARS_CoV_P2']

    for Name in GISAID_SARS_CoV_2:
        # pass
        seq_file = path + 'other_virus/other_virus_seq/seq_GISAID_' + Name + '.csv'
        featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.80)
        featureDF.to_csv(path + 'other_virus/other_virus_Seq_appearance/feature_' + Name + '.csv')

    for Name in NCBI_SARS_CoV_2:
        # pass
        seq_file = path + 'other_virus/other_virus_seq/seq_NCBI_' + Name + '.csv'
        featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.80)
        featureDF.to_csv(path + 'other_virus/other_virus_Seq_appearance/feature_' + Name + '.csv')

    for Name in GISAID_other_virusList:
        seq_file = path + 'other_virus/other_virus_seq/seq_GISAID_' + Name + '.csv'
        featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.99)
        featureDF.to_csv(path + 'other_virus/other_virus_Seq_appearance/feature_' + Name + '.csv')

    for Name in NCBI_other_virusList:
        seq_file = path + 'other_virus/other_virus_seq/seq_NCBI_' + Name + '.csv'
        featureDF = get_appearance(feature_file, seq_file, high_type=0, accuracy=0.99)
        featureDF.to_csv(path + 'other_virus/other_virus_Seq_appearance/feature_' + Name + '.csv')


def common_other_virus_FeatureAppearance(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'

    files = glob.glob(path + 'other_virus/other_virus_Seq_appearance/*')

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

    Appearance_DF = Appearance_DF[['feature_HomoSapiens', 'feature_SARS_CoV_2', 'feature_MERS_CoV', 'feature_SARS_CoV_P2',
                                   'feature_HCoV_OC43', 'feature_HCoV_NL63', 'feature_HCoV_229E', 'feature_HCoV_HKU1', 'feature_HAstV_BF34',
                               'feature_SARS_CoV_HKU_39849', 'feature_SARS_CoV_GDH_BJH01',
                               'feature_Manis_javanica', 'feature_Rhinolophus_affinis',
                               'feature_Rhinolophus_bat', 'feature_Canis', 'feature_Felis_catus']]

    Appearance_DF = Appearance_DF.sort_values(by=['feature_HomoSapiens', 'feature_SARS_CoV_2'], ascending=[False, False])
    Appearance_DF.to_csv(path + 'result/other_virus_Appearance_DataFrame.csv')

    col_list = list(Appearance_DF.iloc[:, 2:11])
    Appearance_DF['feature_NCBI_Other_Taxa'] = Appearance_DF[col_list].sum(axis=1)
    Appearance_DF = Appearance_DF.drop(columns=col_list)
    Appearance_DF = Appearance_DF[['feature_HomoSapiens', 'feature_SARS_CoV_2', 'feature_NCBI_Other_Taxa',
                                   'feature_Manis_javanica', 'feature_Rhinolophus_affinis',
                                   'feature_Rhinolophus_bat', 'feature_Canis', 'feature_Felis_catus']]
    Appearance_DF.to_csv(path + 'result/other_virus_Appearance_DataFrame_simplify.csv')



if __name__ == '__main__':
    start = time.time()
    path = './Sars-Cov-2 Project/v3_Dataset/'

    get_select_feature(path)
    calculate_other_virus_Appearance(path)
    common_other_virus_FeatureAppearance(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print(' ')
    print(' ')
    print(' ')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))
