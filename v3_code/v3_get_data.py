import numpy as np
import pandas as pd
from main_Functions import *
import glob
from sklearn.model_selection import StratifiedKFold
import time

def get_data(path):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    print('------------------ Processing get_data ------------------')
    variant_list = ['alpha', 'beta', 'gamma', 'delta']

    for Name in variant_list:
        print('\nNow running the --- {} --- variant\n'.format(Name))
        files = glob.glob(path + 'Variant_virus/GISAID_' + Name + '/*')
        seqName_list, seq_list = [], []
        file_count, data_count = 0, 0
        for file in files:
            file_count += 1
            for seqName, seq in readFASTA_iter(file):
                data_count += 1
                seqName_list.append(seqName)
                seq_list.append(seq)
                if data_count % 1000 == 0:
                    print('{} ... No.{} file with {} sequence'.format(Name, file_count, data_count))

        print('\n{} : {} sequence\n'.format(Name, len(seqName_list)))
        print('Saving the seqName_list...\n')
        pd.DataFrame(seqName_list).to_csv(path + 'Seq_and_SeqName/seqName_GISAID_' + Name + '.csv',
            header=None, index=None)
        print('\nSaving the seq_list...')
        pd.DataFrame(seq_list).to_csv(path + 'Seq_and_SeqName/seq_GISAID_' + Name + '.csv',
            header=None, index=None)

def mixed_data(path, delta_num, other_percent=1):
    # path = './Sars-Cov-2 Project/v3_Dataset/'
    variant_list = ['alpha', 'beta', 'gamma', 'delta']

    vectorSize = 0
    delta_num = delta_num * 2
    mix_seqName = []
    mix_sequence = []
    mix_label = []

    for Name in variant_list:
        seqName_file = path + 'Seq_and_SeqName/seqName_GISAID_' + Name + '.csv'
        seqName = pd.read_csv(seqName_file, header=None).values.ravel()  # get the sequence name

        seq_file = path + 'Seq_and_SeqName/seq_GISAID_' + Name + '.csv'
        sequence = pd.read_csv(seq_file, header=None).values.ravel()  # get the sequence


        for i in range(len(sequence)):
            if len(sequence[i]) > vectorSize:
                vectorSize = len(sequence[i])
        print('{}    vector size : {}'.format(Name, vectorSize))

        rand = np.random.randint(100000)

        if Name == 'alpha':
            label = 1
            np.random.seed(rand)
            np.random.shuffle(seqName)
            mix_seqName.append(seqName[:int(delta_num * other_percent)])

            np.random.seed(rand)
            np.random.shuffle(sequence)
            mix_sequence.append(sequence[:int(delta_num * other_percent)])

            seq_labels = np.array([label for x in range(len(seqName[:int(delta_num * other_percent)]))])
            np.random.seed(rand)
            np.random.shuffle(seq_labels)
            mix_label.append(seq_labels)

        elif Name == 'beta':
            label = 2
            np.random.seed(rand)
            np.random.shuffle(seqName)
            mix_seqName.append(seqName[:int(delta_num * other_percent)])

            np.random.seed(rand)
            np.random.shuffle(sequence)
            mix_sequence.append(sequence[:int(delta_num * other_percent)])

            seq_labels = np.array([label for x in range(len(seqName[:int(delta_num * other_percent)]))])
            np.random.seed(rand)
            np.random.shuffle(seq_labels)
            mix_label.append(seq_labels)

        elif Name == 'gamma':
            label = 3
            np.random.seed(rand)
            np.random.shuffle(seqName)
            mix_seqName.append(seqName[:int(delta_num * other_percent)])

            np.random.seed(rand)
            np.random.shuffle(sequence)
            mix_sequence.append(sequence[:int(delta_num * other_percent)])

            seq_labels = np.array([label for x in range(len(seqName[:int(delta_num * other_percent)]))])
            np.random.seed(rand)
            np.random.shuffle(seq_labels)
            mix_label.append(seq_labels)

        elif Name == 'delta':
            label = 0
            np.random.seed(rand)
            np.random.shuffle(seqName)
            mix_seqName.append(seqName[:delta_num])

            np.random.seed(rand)
            np.random.shuffle(sequence)
            mix_sequence.append(sequence[:delta_num])

            seq_labels = np.array([label for x in range(len(seqName[:delta_num]))])
            np.random.seed(rand)
            np.random.shuffle(seq_labels)
            mix_label.append(seq_labels)

    pd.DataFrame(mix_seqName).to_csv(path + 'mix_seqName.csv', header=None, index=None)
    pd.DataFrame(mix_sequence).to_csv(path + 'mix_mix_sequence.csv', header=None, index=None)
    pd.DataFrame(mix_label).to_csv(path + 'mix_label.csv', header=None, index=None)

    return vectorSize

def train_valid_data(path, n_splits=2):
    # path = './Sars-Cov-2 Project/v3_Dataset/'

    mix_seqName = pd.read_csv(path + 'mix_seqName.csv', header=None).values.ravel()
    mix_sequence = pd.read_csv(path + 'mix_mix_sequence.csv', header=None).values.ravel()
    mix_label = pd.read_csv(path + 'mix_label.csv', header=None).values.ravel()

    skf = StratifiedKFold(n_splits=n_splits)
    # skf.get_n_splits(mix_sequence, mix_label)
    for train_index, test_index in skf.split(mix_sequence, mix_label):
        mix_sequence_train, mix_sequence_test = mix_sequence[train_index], mix_sequence[test_index]
        mix_seqName_train, mix_seqName_test = mix_seqName[train_index], mix_seqName[test_index]
        mix_label_train, mix_label_test = mix_label[train_index], mix_label[test_index]

    pd.DataFrame(mix_sequence_train).to_csv(path + 'index/train_sequence.csv', header=None, index=None)
    pd.DataFrame(mix_sequence_test).to_csv(path + 'index/valid_sequence.csv', header=None, index=None)

    pd.DataFrame(mix_seqName_train).to_csv(path + 'index/train_seqName.csv', header=None, index=None)
    pd.DataFrame(mix_seqName_test).to_csv(path + 'index/valid_seqName.csv', header=None, index=None)

    pd.DataFrame(mix_label_train).to_csv(path + 'index/train_label.csv', header=None, index=None)
    pd.DataFrame(mix_label_test).to_csv(path + 'index/valid_label.csv', header=None, index=None)


if __name__ == '__main__':
    start = time.time()
    path = './Sars-Cov-2 Project/v3_Dataset/'

    get_data(path) # Run only once
    vectorSize = mixed_data(path, delta_num=2000, other_percent=1)
    train_valid_data(path, n_splits=2)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print(' ')
    print(' ')
    print(' ')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))



