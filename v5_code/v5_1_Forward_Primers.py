import numpy as np
import pandas as pd
from main_Functions import *
from collections import Counter
import glob
import time


def get_forward_primers(path):
    # path = './Sars-Cov-2 Project/v5_Dataset/'
    print('------------------ Processing get_data ------------------')

    files = glob.glob(path + 'forward_primer/*')
    forward_primer_list = []
    count = 0
    print('Start to collect forward primers...')
    for file in files:
        primers = pd.read_csv(file)['Unnamed: 0'].values.ravel()
        for forward_primer in primers:
            count += 1
            forward_primer_list.append(forward_primer)
    print('Saving the forward primers...'.format(len(forward_primer_list), count))
    forward_primer_list = list(set(forward_primer_list))
    print('Finished !     ({}/{})\n'.format(len(forward_primer_list), count))
    pd.DataFrame(forward_primer_list).to_csv(path + 'forward_primer.csv', header=None, index=None)

def CG_content_chect(path, min_CG=0.35, max_CG=0.65):
    # path = './Sars-Cov-2 Project/v5_Dataset/'
    print('Checking the CG content of the forward primers obtained....')

    features = pd.read_csv(path + 'forward_primer.csv', header=None).values.ravel()

    new_features = []
    for feature in features:
        if 'C' in feature and 'G' in feature:
            CountFeature_dict = Counter(feature)
            CG_content = CountFeature_dict.get('C') + CountFeature_dict.get('G')

            if len(feature) * min_CG < CG_content < len(feature) * max_CG:
                new_features.append(feature)
        else:
            pass

    Feature_List = list(set(new_features))
    print('Finished !     ({}/{})        min={}  max={}'.format(len(Feature_List), len(features), min_CG, max_CG))
    pd.DataFrame(Feature_List).to_csv(path + 'forward_primer_CG_check.csv', header=None, index=None)

def get_after_primer_data(path):
    # path = './Sars-Cov-2 Project/v5_Dataset/'

    seq_file = path + 'Seq_and_SeqName/seq_GISAID_delta.csv'
    sequence = pd.read_csv(seq_file, header=None).values.ravel()  # get the sequence
    primers = pd.read_csv(path + 'forward_primer.csv', header=None).values.ravel()

    count = 0
    for forward_primer in primers:
        count += 1
        print('\n{}/{}  Now processing with forward primer:  {}\n'.format(count, len(primers), forward_primer))
        in_count, out_count = 0, 0
        second_half_list = []
        for i in range(len(sequence)):
            if i % 5000 == 0:
                print('No. {} sequence... with {} in the sequence     /     with {} out of the sequence'.format(i, in_count, out_count))
            if forward_primer in sequence[i]:
                in_count += 1
                second_half = sequence[i].split(forward_primer)[1]
                second_half_list.append(second_half)
            else:
                out_count += 1
                pass
        print('\n{} : {} sequence\n'.format(forward_primer, len(second_half_list)))
        pd.DataFrame(second_half_list).to_csv(path + 'second_data/' + str(forward_primer) + '.csv', header=None, index=None)


if __name__ == '__main__':
    start = time.time()

    path = './Sars-Cov-2 Project/v5_Dataset/'
    get_forward_primers(path)
    CG_content_chect(path, min_CG=0.5, max_CG=0.6)
    get_after_primer_data(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print('\n\n')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))
