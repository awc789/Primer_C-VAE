from main_Functions import *
from Main_model_Functions import *
from v3_get_data import *
from v3_filter import *
from v3_feature import *
from v3_appearance import *
from v3_other_appearance import *

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import random
import time

if __name__ == '__main__':
    start = time.time()
    path = './Sars-Cov-2 Project/v3_Dataset/'

    get_data(path) # Run only once
    vectorSize = mixed_data(path, delta_num=2000, other_percent=1)
    train_valid_data(path, n_splits=2)

    variant = ['alpha', 'beta', 'gamma', 'delta']

    seq_T = pd.read_csv(path + 'index/train_sequence.csv', header=None).values.ravel()
    label_T = pd.read_csv(path + 'index/train_label.csv', header=None).values.ravel()

    seq_V = pd.read_csv(path + 'index/valid_sequence.csv', header=None).values.ravel()
    label_V = pd.read_csv(path + 'index/valid_label.csv', header=None).values.ravel()

    rand = np.random.randint(100000)
    np.random.seed(rand)
    np.random.shuffle(seq_T)
    np.random.seed(rand)
    np.random.shuffle(label_T)

    rand = np.random.randint(100000)
    np.random.seed(rand)
    np.random.shuffle(seq_V)
    np.random.seed(rand)
    np.random.shuffle(label_V)
    '----------------------------------------------------------------------------------------------------------'

    f = open(path + 'model/outputVector.txt', 'w')
    f.write('1\n')
    f.write('1\n')

    # Parameters
    kfoldIndex = 0
    batchSize = 50
    vectorSize = 31079
    labelSize = 4  # int(np.max(label_T) + 1)
    beta = 0.001
    limit = 1.01
    iterMax = 50
    # ------------------------------------------------
    w1 = 12
    wd1 = 21
    h1 = 148  # vectorSize/h1 ~ 210 ---> 31079/148 ~ 210
    w4 = 256
    # ------------------------------------------------
    # initialize variables
    iter = 0
    train_accuracy = 0.0
    valid_accuracy = 0.0
    test_accuracy = 0.0
    # best validation accuracy
    best = 0
    validWindow = [0, 0, 0]
    repeatWindow = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    validBest = 1e6
    yResult = []
    yTest = []

    oneHot_labels_T = oneHot(label_T, labelSize)
    oneHot_labels_V = oneHot(label_V, labelSize)
    runs = int(len(oneHot_labels_T) / batchSize)

    '----------------------------------------------------------------------------------------------------------'

    # Tensorflow CNN model
    print('\nStrat to build the CNN model...')
    tf.disable_v2_behavior()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, vectorSize])  # input variable
    keep_prob = tf.placeholder(tf.float32)  # keep between 0.50 to 1.0
    y_ = tf.placeholder(tf.float32, [None, labelSize])  # expected outputs variable
    x_image0 = tf.reshape(x, [-1, 1, vectorSize, 1])  # arrange the tensor as an image (1*30145) 1 channel
    x_image = tf.transpose(x_image0, perm=[0, 3, 2, 1])  # arrange the tensor into 1 channels (1*30145)

    # 1 LAYER
    W_conv1 = weight_variable([1, wd1, 1, w1])
    b_conv1 = bias_variable([w1])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, h1, 1], strides=[1, 1, h1, 1], padding='SAME')

    # Rectifier LAYER
    coef = int(h_pool1.get_shape()[1] * h_pool1.get_shape()[2] * h_pool1.get_shape()[3])  # 1 * 209.34 * 12 ~ 2512
    h_pool2_flat = tf.reshape(h_pool1, [-1, coef])
    W_fc1 = weight_variable([coef, w4])
    b_fc1 = bias_variable([w4])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Rectifier-Dropout LAYER
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([w4, labelSize])
    b_fc2 = bias_variable([labelSize])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Loss Function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_) + beta * tf.nn.l2_loss(W_conv1))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    trueResult = tf.argmax(y_conv, 1)
    trueTest = tf.argmax(y_, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('\nFinished the model building!')

    '----------------------------------------------------------------------------------------------------------'

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    while (best < limit) & (iter < iterMax):
        indexBatch = []
        for iB in range(0, len(oneHot_labels_T)):
            indexBatch.append(iB)
            random.shuffle(indexBatch)
        for run in range(0, runs):
            xa, ya = getBatch_run(seq_T, oneHot_labels_T, batchSize, run, indexBatch, vectorSize)
            train_step.run(feed_dict={x: xa, y_: ya, keep_prob: 0.5})

        xa, ya = getBatch(seq_T, oneHot_labels_T, batchSize, vectorSize)
        train_accuracy = accuracy.eval(feed_dict={x: xa, y_: ya, keep_prob: 1.0})

        xaV, yaV = getBatch(seq_V, oneHot_labels_V, oneHot_labels_V.shape[0], vectorSize)
        valid_accuracy = accuracy.eval(feed_dict={x: xaV, y_: yaV, keep_prob: 1.0})
        cross_entropyVal = cross_entropy.eval(feed_dict={x: xaV, y_: yaV, keep_prob: 1.0})
        cross_entropyTrain = cross_entropy.eval(feed_dict={x: xa, y_: ya, keep_prob: 1.0})

        validWindowValue = 0
        tempValid = validWindow
        for i in range(0, len(validWindow) - 1):
            tempValid[i] = validWindow[i + 1]
        for i in range(0, len(validWindow)):
            validWindow[i] = tempValid[i]
        validWindow[len(validWindow) - 1] = valid_accuracy
        for i in range(0, len(validWindow)):
            validWindowValue = validWindowValue + validWindow[i]
        validWindowValue = validWindowValue / len(validWindow)
        tempValid = repeatWindow
        for i in range(0, len(repeatWindow) - 1):
            tempValid[i] = repeatWindow[i + 1]
        for i in range(0, len(repeatWindow)):
            repeatWindow[i] = tempValid[i]
        repeatWindow[len(repeatWindow) - 1] = valid_accuracy
        if np.var(repeatWindow) == 0 and iter > iterMax:
            iter = iter
        if validWindowValue > best or cross_entropyVal < validBest:
            validBest = cross_entropyVal
            best = validWindowValue

            xaT, yaT = getBatch(seq_V, oneHot_labels_V, oneHot_labels_V.shape[0], vectorSize)
            test_accuracy = accuracy.eval(feed_dict={x: xaT, y_: yaT, keep_prob: 1.0})
            if kfoldIndex == 0:
                save_path = saver.save(sess, path + "model/CNN_model.ckpt")

            results = correct_prediction.eval(feed_dict={x: xaT, y_: yaT, keep_prob: 1.0})
            yResult = trueResult.eval(feed_dict={x: xaT, y_: yaT, keep_prob: 1.0})
            yTest = trueTest.eval(feed_dict={x: xaT, y_: yaT, keep_prob: 1.0})

            fOut = open(path + 'model/outputVector_1.txt', 'w')
            fOut.write('1\n')
            fOut.write('1\n')
            temp = 1.0 - best
            trueAcc = str(temp)
            print(trueAcc)
            fOut.write(trueAcc + '\n')
            fOut.close()
        log = "%d	%d	%g	%g	%g	%g	%g	%g" % (iter, kfoldIndex, train_accuracy, valid_accuracy, best,
                                                                test_accuracy, cross_entropyVal, cross_entropyTrain)
        print(log)
        f.write(log + '\n')
        iter = iter + 1

    f.close()
    np.savetxt(path + 'model/results.txt', yResult, fmt='%i', delimiter=' ')
    np.savetxt(path + 'model/test.txt', yTest, fmt='%i', delimiter=' ')
    f = open(path + 'model/log3.txt', 'a')
    name = '2021-08-06 / CNN model'
    f.write(name + '\n')
    f.close()

    f = open(path + 'model/outputVector_2.txt', 'w')
    f.write('1\n')
    f.write('1\n')
    temp = 1.0 - best
    trueAcc = str(temp)
    print(trueAcc)
    f.write(trueAcc + '\n')
    f.close()

    xaT, yaT = getBatch(seq_V, oneHot_labels_V, oneHot_labels_V.shape[0], vectorSize)

    units = sess.run(h_conv1, feed_dict={x: xaT, y_: yaT, keep_prob: 1.0})
    print(units.shape)
    units = sess.run(h_pool1, feed_dict={x: xaT, y_: yaT, keep_prob: 1.0})
    print(units.shape)

    '----------------------------------------------------------------------------------------------------------'

    seqName_file = path + 'Seq_and_SeqName/seqName_GISAID_' + variant[3] + '.csv'
    seqName = pd.read_csv(seqName_file, header=None).values.ravel()  # get the sequence name

    # seq_file = path + 'NCBI_Pangolin/NCBi_2000.fasta'
    seq_file = path + 'Seq_and_SeqName/seq_GISAID_' + variant[3] + '.csv'
    sequence = pd.read_csv(seq_file, header=None).values.ravel()  # get the sequence

    np.random.shuffle(seqName)
    np.random.shuffle(sequence)
    seqName = seqName[:1000]
    sequence = sequence[:1000]
    seq_labels = np.array([0 for x in range(len(sequence))])

    seq_count = 0
    outData, outLabels = [], []
    print('\nStrat to transfer the sequence ...\n')
    for i in range(len(sequence)):
        seq_count += 1
        if seq_count % 100 == 0:
            print('Transferring...  No.{} with total {}'.format(seq_count, len(seq_labels)))
        sample = np.zeros(vectorSize)
        for j in range(0, len(sequence[i])):
            if sequence[i][j] == 'C':
                sample[j] = 0.25
            elif sequence[i][j] == 'T':
                sample[j] = 0.50
            elif sequence[i][j] == 'G':
                sample[j] = 0.75
            elif sequence[i][j] == 'A':
                sample[j] = 1.0
            else:
                sample[j] = 0.0
        outData.append(sample)
        outLabels.append(seq_labels[i])

    print('\nTransfer finished!\n')
    pd.DataFrame(sequence).to_csv(path + 'filter_seq/filter_seq.csv', header=None, index=None)
    data = np.array(outData)
    seq_labels = np.array(outLabels)
    oneHotLabels = oneHot(seq_labels, labelSize)

    '----------------------------------------------------------------------------------------------------------'
    'Filter files'

    print('the Filter files')
    units = sess.run(h_conv1, feed_dict={x: data, y_: oneHotLabels, keep_prob: 1.0})
    print('\nThe first h_conv1 layer size:      h_conv1 = {}\n'.format(units.shape))

    sampleSize = int(data.shape[0])
    Mat = np.zeros((sampleSize, vectorSize))

    for filterIndex in range(units.shape[3]):
        print('Loop 1 : Generating the  {}  Filter file'.format(filterIndex))
        for testSize in range(sampleSize):
            for inputSize in range(vectorSize):
                Mat[testSize][inputSize] = units[testSize][0][inputSize][filterIndex]
        pd.DataFrame(Mat).to_csv(path + '/filter/filter_' + str(filterIndex) + '.csv', header=None, index=None)

    sess.close()

    posPool(path)
    creatFeatVector(path)
    getFeature(path)

    sameFeature(path)
    calculateAppearance(path)
    commonFeatureAppearance(path)

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
