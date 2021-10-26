import numpy as np
import pandas as pd
import primer3
from collections import Counter
import glob
import time

# def primer_start_and_end_check(path):
#     # path = './Sars-Cov-2 Project/v5_Dataset/'


if __name__ == '__main__':
    path = './Sars-Cov-2 Project/v5_Dataset/'

    primers = pd.read_csv(path + 'amplicon_length/r_new_primers.csv')
    drop_list = []

    for i in range(len(primers)):
        temp = primers[i:i + 1]
        forward_primer = temp['Forward primer'][i]
        reverse_primer = temp['Reverse primer'][i]

        if 'C' not in forward_primer[-3:] and 'G' not in forward_primer[-3:]:
            drop_list.append(i)
        else:
            if 'C' not in reverse_primer[-3:] and 'G' not in reverse_primer[-3:]:
                drop_list.append(i)
            else:
                if primer3.calcHomodimer(forward_primer).dg <= -9000 or primer3.calcHomodimer(reverse_primer).dg <= -9000:
                    drop_list.append(i)
                else:
                    if primer3.calcHeterodimer(forward_primer, reverse_primer).dg <= -9000:
                        drop_list.append(i)
                    else:
                        if abs(primer3.calcTm(forward_primer) - primer3.calcTm(reverse_primer)) > 5:
                            drop_list.append(i)
                        else:
                            if primer3.calcTm(forward_primer) > 60 or primer3.calcTm(reverse_primer) < 50:
                                drop_list.append(i)
                            else:
                                if primer3.calcTm(reverse_primer) > 60 or primer3.calcTm(reverse_primer) < 50:
                                    drop_list.append(i)
                                else:
                                    primers.iloc[i:i + 1, 4] = primer3.calcTm(forward_primer)
                                    primers.iloc[i:i + 1, 5] = primer3.calcTm(reverse_primer)
                                    primers.iloc[i:i + 1, 6] = abs(primer3.calcTm(forward_primer) - primer3.calcTm(reverse_primer))


    primers = primers.drop(index=drop_list)
    primers.to_csv(path + 'amplicon_length/r_new_primers_deltaG.csv', index=None)


