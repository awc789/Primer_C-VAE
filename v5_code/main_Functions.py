def readFASTA(fa):
    '''
    :msg: read a xxx.fasta file
    :param fa: --- {str} --- the path of the xxx.fasta file
    :return: --- {dict} --- return a dictionary with key = seqName and value = sequence
    '''
    FA = open(fa)
    seqDict = {}
    for line in FA:
        if line.startswith('>'):
            seqName = line.replace('>', '').split()[0]
            seqDict[seqName] = ''
        else:
            seqDict[seqName] += line.replace('\n', '').strip()
    FA.close()
    return seqDict


def readFASTA_iter(fa):
    '''
    :msg: read a xxx.fasta file
    :param fa: --- {str} --- the path of the xxx.fasta file
    :return: --- {generator} --- return a generator which gives each sequence name and sequence of the xxx.fasta file
    '''
    with open(fa, 'r') as FA:
        seqName, seq = '', ''
        while 1:
            line = FA.readline()
            line = line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield((seqName, seq))
            if line.startswith('>'):
                seqName = line[1:].split()[0]
                seq = ''
            else:
                seq += line
            if not line:
                break


def getSeq(fa, querySeqName, start=1, end=0):
    '''
    :msg: get a particular sequence of a xxx.fasta file
    :param fa: --- {str} --- the path of the xxx.fasta file
    :param querySeqName: --- {str} --- the name of the particular sequence
    :param start: --- {int} --- the starting position of intercepting the sequence (defaults to 1)
    :param end: --- {int} --- the ending position of intercepting the sequence (defaults to 0 / full length)
    :return: --- {str} --- the sequence which intercepted
    '''
    if start < 0:
        start = start + 1
    for seqName, seq in readFASTA_iter(fa):
        if querySeqName == seqName:
            if end != 0:
                returnSeq = seq[start-1: end]
                print('The start position and end position is {} / {}'.format(start-1, end))
            else:
                returnSeq = seq[start-1: ]
            return returnSeq


def getReverseComplement(sequence):
    '''
    :msg: get the reverse cDNA of the RNA sequence
    :param sequence: --- {str} --- a RNA sequence of the virus
    :return: --- {str} --- the reverse cDNA sequence of the given RNA
    '''
    sequence = sequence.upper()
    sequence = sequence.replace('A', 't')
    sequence = sequence.replace('T', 'a')
    sequence = sequence.replace('C', 'g')
    sequence = sequence.replace('G', 'c')
    return sequence.upper()[::-1]


def getGC(sequence):
    '''
    :msg: get the GC content of a sequence
    :param sequence: --- {str} --- a sequence of RNA
    :return: --- {float} --- the GC content of sequence
    '''
    sequence = sequence.upper()
    content = (sequence.count("G") + sequence.count("C")) / len(sequence)
    return content


def readSeqByWindow(sequence, winSize, stepSize):
    '''
    :msg: sliding window to read a sequence
    :param sequence: --- {str} --- a sequence of RNA
    :param winSize: --- {int} --- the Window size
    :param stepSize: --- {int} --- the Step size
    :return: --- {generator} --- return a generator which gives each sequence of the Window
    '''
    if stepSize <= 0:
        return False
    now = 0
    seqLen = len(sequence)
    while (now + winSize - stepSize < seqLen):
        yield sequence[now:now + winSize]
        now += stepSize


