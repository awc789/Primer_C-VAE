import numpy as np
import pandas as pd
from collections import Counter
import glob
import time

def reverse(path):
    # path = './Sars-Cov-2 Project/v5_Dataset/'
    new_reverse_primer = []
    new_Reverse_Melting_Temperature = []

    '''--------------------------------------------------------------------------------------------------------------'''
    print('(1/2) Exist primers: ')
    Final_primers = pd.read_csv(path + 'amplicon_length/exist_primers.csv')
    reverse_primer = Final_primers['Reverse primer'].values.ravel()
    for i in range(len(reverse_primer)):
        temp = list(reverse_primer[i])
        new_primer = []
        for j in range(len(temp)):
            if temp[len(temp) - j - 1] == 'A':
                new_primer.append('T')
            elif temp[len(temp) - j - 1] == 'T':
                new_primer.append('A')
            elif temp[len(temp) - j - 1] == 'G':
                new_primer.append('C')
            elif temp[len(temp) - j - 1] == 'C':
                new_primer.append('G')

        if len(set(new_primer)) == 4:
            counter = Counter(new_primer)
            Tm = 64.9 + 41 * (counter.get('G') + counter.get('C') - 16.4) / (counter.get('A') + counter.get('T') +
                                                                             counter.get('G') + counter.get('C'))
        else:
            Tm = -1

        var = ''.join(new_primer)
        new_reverse_primer.append(var)
        new_Reverse_Melting_Temperature.append(Tm)

    Final_primers['Reverse primer'] = new_reverse_primer
    Final_primers['Reverse Melting Temperature (Tm)'] = new_Reverse_Melting_Temperature
    Final_primers['Tm difference'] = abs(Final_primers['Forward Melting Temperature (Tm)'] -
                                         Final_primers['Reverse Melting Temperature (Tm)'])
    Final_primers.to_csv(path + 'amplicon_length/r_exist_primers.csv', index=None)
    print('Finished !')

    '''--------------------------------------------------------------------------------------------------------------'''
    new_reverse_primer = []
    new_Reverse_Melting_Temperature = []

    print('\n(2/2) new primers')
    Final_primers = pd.read_csv(path + 'amplicon_length/new_primers.csv')
    reverse_primer = Final_primers['Reverse primer'].values.ravel()
    for i in range(len(reverse_primer)):
        temp = list(reverse_primer[i])
        new_primer = []
        for j in range(len(temp)):
            if temp[len(temp) - j - 1] == 'A':
                new_primer.append('T')
            elif temp[len(temp) - j - 1] == 'T':
                new_primer.append('A')
            elif temp[len(temp) - j - 1] == 'G':
                new_primer.append('C')
            elif temp[len(temp) - j - 1] == 'C':
                new_primer.append('G')

        if len(set(new_primer)) == 4:
            counter = Counter(new_primer)
            Tm = 64.9 + 41 * (counter.get('G') + counter.get('C') - 16.4) / (counter.get('A') + counter.get('T') +
                                                                             counter.get('G') + counter.get('C'))
        else:
            Tm = -1

        var = ''.join(new_primer)
        new_reverse_primer.append(var)
        new_Reverse_Melting_Temperature.append(Tm)

    Final_primers['Reverse primer'] = new_reverse_primer
    Final_primers['Reverse Melting Temperature (Tm)'] = new_Reverse_Melting_Temperature
    Final_primers['Tm difference'] = abs(Final_primers['Forward Melting Temperature (Tm)'] -
                                         Final_primers['Reverse Melting Temperature (Tm)'])
    Final_primers.to_csv(path + 'amplicon_length/r_new_primers.csv', index=None)
    print('Finished !')


if __name__ == '__main__':
    start = time.time()

    path = './Sars-Cov-2 Project/v5_Dataset/'
    reverse(path)

    elapsed = (time.time() - start)
    hour = int(elapsed // 3600)
    minute = int((elapsed % 3600) // 60)
    second = int(elapsed - 3600 * hour - 60 * minute)
    print('\n\n')
    print("Time used:  {} hours {} minutes {} seconds  ---->  all  {} seconds".format(hour, minute, second, elapsed))