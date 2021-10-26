import numpy as np
import pandas as pd
from main_Functions import *
import glob
import time


def exist_primer_check(path):
    # path = './Sars-Cov-2 Project/v5_Dataset/'

    primers = pd.read_csv(path + 'forward_primer_CG_check_Homo_Dimer_check.csv', header=None).values.ravel()
    count = 0
    print('Start the the exist primer check...        {} in total\n'.format(len(primers)))

    for forward_primer in primers:
        count += 1
        reverse_primers = {}
        print('No.{} file, the forward primer is    {}'.format(count, forward_primer))

        sequence = pd.read_csv(path + 'second_data/' + forward_primer + '.csv', header=None).values.ravel()
        min_num = int(len(sequence) * 0.99)
        sequence = pd.DataFrame(sequence)
        p = 0

        for primer in primers:
            if forward_primer == primer:
                pass
            else:
                count_feature = sequence[0].apply(lambda x: x.count(primer))
                available_count = len(count_feature[count_feature == 1])
                if available_count >= min_num:
                    p = 1
                    reverse_primers[primer] = available_count / len(sequence)

        if p == 1:
            print('Yes')
            reverse_primers = pd.Series(reverse_primers).to_frame()
            reverse_primers.to_csv(path + 'exist_primer_check/' + forward_primer + '.csv')
        else:
            print('No')
    print('\nComplete the exist primer check !\n')


if __name__ == '__main__':
    start = time.time()

    path = './Sars-Cov-2 Project/v5_Dataset/'
    exist_primer_check(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print('\n\n')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))