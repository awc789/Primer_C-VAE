import numpy as np
import pandas as pd
from collections import Counter
import glob
import os
import shutil
import time


def get_available_primers(path):
    # path = './Sars-Cov-2 Project/v4_Dataset/'
    print('Get available primers with 40%-60% CG content...')
    if os.path.exists(path + 'result/temp_feature.csv'):
        os.remove(path + 'result/temp_feature.csv')

    features = pd.read_csv(path + 'forward_primer_CG_check_Homo_Dimer_check.csv', header=None).values.ravel()

    files = glob.glob(path + 'result/*')
    for file in files:
        forward_primer = file.split('/')[-1].split('_')[0]
        if forward_primer in features:
            newpath = path + 'CG_Check/new_primers/' + forward_primer + '.csv'
            shutil.copyfile(file, newpath)

    files = glob.glob(path + 'exist_primer_check/*')
    for file in files:
        forward_primer = file.split('/')[-1].split('.')[0]
        if forward_primer in features:
            newpath = path + 'CG_Check/exist_primers/' + forward_primer + '.csv'
            shutil.copyfile(file, newpath)

    print('Finished ! \n')


def calculate_CG_content_and_melting_temperature(primers):
    primers = list(primers)
    if len(set(primers)) == 4:
        counter = Counter(primers)
        Tm = 64.9 + 41 * (counter.get('G') + counter.get('C') - 16.4) / (counter.get('A') + counter.get('T') +
                                                                         counter.get('G') + counter.get('C'))
        CG_content = (counter.get('C') + counter.get('G')) / len(primers)
        return CG_content, Tm
    else:
        return -1, -1


def length_of_amplicon(path):
    # path = './Sars-Cov-2 Project/v4_Dataset/'

    print('Get the length of amplicon between the primers...\n')
    feature_CG_Check = pd.read_csv(path + 'forward_primer_CG_check_Homo_Dimer_check.csv', header=None).values.ravel()
    primer_design = []

    '''--------------------------------------------------------------------------------------------------------------'''
    print('(1/2) Exist primers: ')
    files = glob.glob(path + 'CG_Check/exist_primers/*')
    count = 0

    for file in files:
        count += 1
        forward_primer = file.split('/')[-1].split('.')[0]
        print('No.{} out of {} file:          {}'.format(count, len(files), forward_primer))

        sequence = pd.read_csv(path + 'second_data/' + str(forward_primer) + '.csv', header=None).values.ravel()
        reverse_primers = pd.read_csv(file)['Unnamed: 0'].values.ravel()
        for reverse_primer in reverse_primers:
            length = []
            if reverse_primer in feature_CG_Check:
                for seq in sequence:
                    if reverse_primer in seq:
                        amplicon = len(seq.split(reverse_primer)[0]) + len(forward_primer) + len(reverse_primer)
                        length.append(amplicon)
                    else:
                        pass
            else:
                pass

            if len(length) == 0:
                pass
            else:
                maxx = np.max(length)
                minn = np.min(length)
                mean = np.mean(length)
                f_CG, f_Tm = calculate_CG_content_and_melting_temperature(forward_primer)
                r_CG, r_Tm = calculate_CG_content_and_melting_temperature(reverse_primer)

                primer_design.append(forward_primer)
                primer_design.append(reverse_primer)
                primer_design.append(f_CG)
                primer_design.append(r_CG)
                primer_design.append(f_Tm)
                primer_design.append(r_Tm)
                primer_design.append(abs(f_Tm-r_Tm))

                primer_design.append(mean)
                primer_design.append(maxx)
                primer_design.append(minn)

    pd.DataFrame(primer_design).to_csv(path + 'amplicon_length/exist_primers.csv', header=None, index=None)

    '''--------------------------------------------------------------------------------------------------------------'''
    print('\n(2/2) new primers: ')
    primer_design = []
    files = glob.glob(path + 'CG_Check/new_primers/*')
    count = 0

    for file in files:
        count += 1
        forward_primer = file.split('/')[-1].split('.')[0]
        print('No.{} out of {} file:          {}'.format(count, len(files), forward_primer))

        sequence = pd.read_csv(path + 'second_data/' + str(forward_primer) + '.csv', header=None).values.ravel()
        reverse_primers = pd.read_csv(file)['Unnamed: 0'].values.ravel()
        for reverse_primer in reverse_primers:
            length = []
            for seq in sequence:
                if reverse_primer in seq:
                    amplicon = len(seq.split(reverse_primer)[0]) + len(forward_primer) + len(reverse_primer)
                    length.append(amplicon)
                else:
                    pass

            if len(length) == 0:
                pass
            else:
                maxx = np.max(length)
                minn = np.min(length)
                mean = np.mean(length)
                f_CG, f_Tm = calculate_CG_content_and_melting_temperature(forward_primer)
                r_CG, r_Tm = calculate_CG_content_and_melting_temperature(reverse_primer)

                primer_design.append(forward_primer)
                primer_design.append(reverse_primer)
                primer_design.append(f_CG)
                primer_design.append(r_CG)
                primer_design.append(f_Tm)
                primer_design.append(r_Tm)
                primer_design.append(abs(f_Tm - r_Tm))

                primer_design.append(mean)
                primer_design.append(maxx)
                primer_design.append(minn)

    pd.DataFrame(primer_design).to_csv(path + 'amplicon_length/new_primers.csv', header=None, index=None)


def reshape_file(path):
    # path = './Sars-Cov-2 Project/v4_Dataset/'

    Name_list = ['Forward primer', 'Reverse primer', 'Forward CG content', 'Reverse CG content',
                 'Forward Melting Temperature (Tm)', 'Reverse Melting Temperature (Tm)', 'Tm difference',
                 'amplicon_avg', 'amplicon_max', 'amplicon_min']

    new_primers = pd.read_csv(path + 'amplicon_length/new_primers.csv', header=None).values.ravel()
    new_primers = new_primers.reshape(-1, len(Name_list))
    new_primers = pd.DataFrame(new_primers)
    new_primers.columns = Name_list
    new_primers.to_csv(path + 'amplicon_length/new_primers.csv', index=None)

    exist_primers = pd.read_csv(path + 'amplicon_length/exist_primers.csv', header=None).values.ravel()
    exist_primers = exist_primers.reshape(-1, len(Name_list))
    exist_primers = pd.DataFrame(exist_primers)
    exist_primers.columns = Name_list
    exist_primers.to_csv(path + 'amplicon_length/exist_primers.csv', index=None)



if __name__ == '__main__':
    start = time.time()

    path = './Sars-Cov-2 Project/v5_Dataset/'
    get_available_primers(path)
    length_of_amplicon(path)
    reshape_file(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print('\n\n')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))