import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from collections import Counter
import os
import math
import matplotlib.cbook as cbook
import urllib
import os

from compiler.ast import flatten
import operator

from sklearn.tree import *
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz




# ==============================================================================
#                        PART 1
# ==============================================================================
def read_data(filename):
    filelines = []
    for line in open(filename):
        splited = line.strip("\n").split(" ")
        filelines.append(splited)
    return filelines

def creat_dict(data_lines):
    ddict = {}
    for line in data_lines:
        ext = []
        for w in line:
            if w not in ext:
                ext.append(w)
                if w not in ddict.keys():
                    ddict[w] = 1
                else:
                    ddict[w] += 1

    return ddict

def part1():
    real_data = read_data("./clean_real.txt")
    fake_data = read_data("./clean_fake.txt")

    len_real = len(real_data)
    len_fake = len(fake_data)

    real_dict = creat_dict(real_data)
    fake_dict = creat_dict(fake_data)

    for k in real_dict.keys():
        real_dict[k] = float(real_dict[k])/len_real
    for k in fake_dict.keys():
        fake_dict[k] = float(fake_dict[k])/len_fake

    keys = real_dict.keys()
    keys.append(fake_dict.keys())

    k_diff = []
    for k in keys:
        if k in real_dict.keys():
            real_p = real_dict[k]
        else:
            real_p = 0

        if k in fake_dict.keys():
            fake_p = fake_dict[k]
        else:
            fake_p = 0

        diff = abs(real_p - fake_p)

        k_diff.append([k, str(diff), real_p, fake_p])

    d123 = [k_diff[0][1], k_diff[1][1], k_diff[2][1]]

    print(d123)

    b1 = d123[d123.index(max(d123))]
    d123.remove(max(d123))
    b2 = d123[d123.index(max(d123))]
    d123.remove(max(d123))
    b3 = d123[d123.index(max(d123))]
    d123.remove(max(d123))

    b1, b2, b3, b4, b5, b6 = k_diff[0], k_diff[1], k_diff[2], k_diff[3], k_diff[4], k_diff[5]
    print("b1b2b3", b1, b2, b3, b4, b5, b6)

    for i in range(6, len(k_diff)):
        print("k_diff[i][-1]", k_diff[i][1])
        print("b1[-1]", b1[1])

        if float(k_diff[i][1]) > float(b1[1]):
            print("in1")
            print("pre: ", b1, b2, b3, b4, b5, b6)
            b6 = b5
            b5 = b4
            b4 = b3
            b3 = b2
            b2 = b1
            b1 = k_diff[i]
            print("after: ", b1, b2, b3, b4, b5, b6)

        elif k_diff[i][1] > b2[1]:

            b6 = b5
            b5 = b4
            b4 = b3
            b3 = b2
            b2 = k_diff[i]
        elif k_diff[i][1] > b3[1]:
            b6 = b5
            b5 = b4
            b4 = b3
            b3 = k_diff[i]

        elif k_diff[i][1] > b4[1]:
            b6 = b5
            b5 = b4
            b4 = k_diff[i]

        elif k_diff[i][1] > b5[1]:
            b6 = b5
            b5 = k_diff[i]

        elif k_diff[i][1] > b6[1]:
            b6 = k_diff[i]

    print(b1, b2, b3, b4, b5, b6)


# ==============================================================================
#                        PART 2
# ==============================================================================

def get_set(data):
    # temp = data.copy()
    temp = data[:]
    np.random.seed(7)
    np.random.shuffle(data)
    length = len(data)
    train = temp[:int(length * 0.7)]
    valid = temp[int(length * 0.7): int(length * 0.85)]
    test = temp[int(length * 0.85):]
    return train, valid, test


def get_label(set, f_or_r):
    length = len(set)
    label = []
    if f_or_r == 1: #real
        label = [1] * length
    else:
        label = [0] * length
    return label


def naive_bayes(count_fake, count_real, m, p_hat, set, f_train, r_train, part3=False):

    estimate = 0
    p_w_class_real = {}
    p_nw_class_real = {}
    p_w_class_fake = {}
    p_nw_class_fake = {}

    len_real = len(r_train)
    len_fake = len(f_train)
    len_total = len_real + len_fake

    log_w_r = []
    log_w_f = []

    for word in set:
        if word in count_real.keys():
            p_w_real = (float(count_real[word]) + m * p_hat) / float(len_real + m)
            # print "word,", word
            # print "count word", count_real[word]
            # print "p(w|real)", p_w_real
            p_w_class_real[word] = p_w_real
            log_w_r.append(math.log(p_w_real))
        else:
            p_nw_real = (float(m * p_hat)) / float(len_real + m)
            # print word
            p_nw_class_real[word] = p_nw_real
            log_w_r.append(math.log(p_nw_real))
        if word in count_fake.keys():
            p_w_fake = (float(count_fake[word]) + m * p_hat) / float(len_fake + m)
            p_w_class_fake[word] = p_w_fake
            log_w_f.append(math.log(p_w_fake))
        else:
            p_nw_fake = (float(m * p_hat)) / float(len_fake + m)
            p_nw_class_fake[word] = p_nw_fake
            log_w_f.append(math.log(p_nw_fake))

    p_real = float(len_real)/len_total
    prob_w_real = math.exp(sum(log_w_r)) * p_real

    p_fake = float(len_fake)/len_total
    prob_w_fake = math.exp(sum(log_w_f)) * p_fake

    if prob_w_fake <= prob_w_real:
        estimate = 1

    if part3 is not True:
        return estimate
    else:
        return p_w_class_real, p_nw_class_real, p_w_class_fake, p_nw_class_fake


def performance2(set, label, count_real, count_fake, m, p_hat, f_train, r_train):
    correct = 0
    for i in range(len(set)):
        # word = set[i].split()
        word = set[i]
        predict = naive_bayes(count_fake, count_real, m, p_hat, word, f_train, r_train)

        # print "//////label i", i, label[i]
        if label[i] == predict:
            # print "here!!!!!!!!!!"
            correct += 1

    return float(correct)/len(label)


def part2():
    real_data = read_data("./clean_real.txt")
    fake_data = read_data("./clean_fake.txt")

    f_train, f_valid, f_test = get_set(fake_data)
    r_train, r_valid, r_test = get_set(real_data)

    count_fake = creat_dict(f_train)
    count_real = creat_dict(r_train)

    r_train_y = get_label(r_train, 1)
    f_train_y = get_label(f_train, 0)

    train_set = f_train + r_train
    train_set_y = f_train_y + r_train_y

    print "train", len(train_set), len(train_set_y)

    # generate label for valid and test
    print "real valid", len(r_valid)
    print "fakr valid", len(f_valid)

    r_valid_y = get_label(r_valid, 1)
    f_valid_y = get_label(f_valid, 0)
    print r_valid_y
    print "real y", len(r_valid_y)
    print "fake y", len(f_valid_y)

    valid_set = f_valid + r_valid
    valid_set_y = f_valid_y + r_valid_y

    print "========valid set", len(valid_set)
    print "========valid y ", len(valid_set_y)

    r_test_y = get_label(r_test, 1)
    f_test_y = get_label(f_test, 0)

    test_set = f_test + r_test
    test_set_y = f_test_y + r_test_y

    print "=== tune m and p ==="
    best_m = 0
    best_p_hat = 0
    best_perform = 0
    mt = np.arange(1, 4, 0.5)
    pt = np.arange(0.1, 0.7, 0.1)
    for m in mt:
        for p_hat in pt:
            print "m: ", m, "p: ", p_hat

            vali_perform = performance2(valid_set, valid_set_y, count_real, count_fake, m, p_hat, f_train, r_train)
            print "performance: ", vali_perform

            if best_perform < vali_perform:
                best_m = m
                best_p_hat = p_hat
                best_perform = vali_perform
    print "=== finish ==="

    perform_train = performance2(train_set, train_set_y, count_real, count_fake, best_m, best_p_hat, f_train, r_train)
    perform_valid = performance2(valid_set, valid_set_y, count_real, count_fake, best_m, best_p_hat, f_train, r_train)
    perform_test = performance2(test_set, test_set_y, count_real, count_fake, best_m, best_p_hat, f_train, r_train)

    print "best m: ", best_m
    print "best p hat: ", best_p_hat
    print "training set performance: ", perform_train
    print "valid set performance:", perform_valid
    print "test set performance:", perform_test

#part2()

# ==============================================================================
#                        PART 3
# ==============================================================================

def part3():
    real_data = read_data("./clean_real.txt")
    fake_data = read_data("./clean_fake.txt")

    f_train, f_valid, f_test = get_set(fake_data)
    r_train, r_valid, r_test = get_set(real_data)

    count_fake = creat_dict(f_train)
    count_real = creat_dict(r_train)

    len_real = len(r_train) + len(r_valid) + len(r_test)
    len_fake = len(f_train) + len(f_valid) + len(f_test)
    len_total = len_real + len_fake

    r_train_y = get_label(r_train, 1)
    f_train_y = get_label(f_train, 0)

    train_set = f_train + r_train
    train_set_y = f_train_y + r_train_y

    # Calculating P(real)
    p_real = float(len_real)/len_total

    # Calculating P(word|class)
    m = 1
    p_hat = 0.3

    training_set = flatten(train_set)
    p_w_class_real, p_nw_class_real, p_w_class_fake, p_nw_class_fake = naive_bayes(count_fake, count_real, m, p_hat, training_set, f_train, r_train, True)


    # ========================= Calculating p(word) ============================
    # P(word) = P(word | fake) P(fake) + P(word | real) P(real)
    p_word = {}
    p_nword = {}

    for word in training_set:
        if word in p_w_class_fake:
            pwf = p_w_class_fake[word]
        else:
            pwf = p_nw_class_fake[word]
        if word in p_w_class_real:
            pwr = p_w_class_real[word]
        else:
            pwr = p_nw_class_real[word]
        prob_word = pwf * (1 - p_real) + pwr * p_real
        prob_nword = (1 - pwf) * (1 - p_real) + (1 - pwr) * p_real

        p_word[word] = prob_word
        p_nword[word] = prob_nword


    # ======================================================================
    # Calculating P(class|word), P(class|not word)
    p_real_word = {}
    p_real_not_word = {}
    p_fake_word = {}
    p_fake_not_word = {}
    for word in p_w_class_real.keys():
        p_real_word[word] = float(p_w_class_real[word] * p_real)/float(p_word[word])
        p_real_not_word[word] = (1 - p_w_class_real[word]) * p_real/float(p_nword[word])

    for word in p_w_class_fake.keys():
        p_fake_word[word] = float(p_w_class_fake[word] * (1 - p_real))/float(p_word[word])
        p_fake_not_word[word] = float((1 - p_w_class_fake[word]) * (1 - p_real))/float(p_nword[word])

    rw_sort = sorted(p_real_word.items(), key = operator.itemgetter(1))

    rnw_sort = sorted(p_real_not_word.items(), key = operator.itemgetter(1))

    fw_sort = sorted(p_fake_word.items(), key = operator.itemgetter(1))

    fnw_sort = sorted(p_fake_not_word.items(), key = operator.itemgetter(1))

    print " ========================= part a ================================= "

    print "10 words whose presence most strongly predicts that the news is real."

    for i in range(-1, -11, -1):
        print "the ", -i ," largest P(real|word) word:", rw_sort[i]

    print "\n10 words whose absence most strongly predicts that the news is real."

    for i in range(-1, -11, -1):
        print "the ", -i ," largest P(real|not word) word:", rnw_sort[i]

    print "\n10 words whose presence most strongly predicts that the news is fake."

    for i in range(-1, -11, -1):
        print "the ", -i ," largest P(fake|word) word:", fw_sort[i]

    print "\n10 words whose absence most strongly predicts that the news is fake."

    for i in range(-1, -11, -1):
        print "the ", -i ," largest P(fake|not word) word:", fnw_sort[i]

    print " ========================= part b ================================= "

    print "10 non-stopwords whose presence most strongly predicts that the news is real."
    count = 0
    for i in range(-1, -len(rw_sort), -1):
        if rw_sort[i][0] not in ENGLISH_STOP_WORDS:
            print "the ", -i ," largest P(fake|word) non-stopwords:", rw_sort[i][0]
            count += 1
        if count == 10:
            break

    print "\n10 non-stopwords whose absence most strongly predicts that the news is real."

    count = 0
    for i in range(-1, -len(rnw_sort), -1):
        if rnw_sort[i][0] not in ENGLISH_STOP_WORDS:
            print "the ", -i ," largest P(fake|not word) non-stopwords:", rnw_sort[i][0]
            count += 1
        if count == 10:
            break

    print "\n10 non-stopwords whose presence most strongly predicts that the news is fake."

    count = 0
    for i in range(-1, -len(fw_sort), -1):
        if fw_sort[i][0] not in ENGLISH_STOP_WORDS:
            print "the ", -i ," largest P(real|not word) non-stopwords:", fw_sort[i][0]
            count += 1
        if count == 10:
            break

    print "\n10 non-stopwords whose absence most strongly predicts that the news is fake."

    count = 0
    for i in range(-1, -len(fnw_sort), -1):
        if fnw_sort[i][0] not in ENGLISH_STOP_WORDS:
            print "the ", -i ," largest P(real|not word) non-stopwords:", fnw_sort[i][0]
            count += 1
        if count == 10:
            break

#part3()

# ==============================================================================
#                        PART 4
# ==============================================================================

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f(x, y, theta):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    p = sigmoid(np.dot(theta.T, x))
    return -sum(y * math.log(p) + (1 - y) * math.log((1 - p)))


def df(x, y, theta, _lambda):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    p = sigmoid(np.dot(theta.T, x))
    return np.dot(x, (p - y).T) + 2 * _lambda * theta


def performance(x, y, theta):

    #  x.shape = (4806, 2285) => 2086 words, 2285 headlines
    # print("=======================================")
    # print("before x shape", x.shape)
    # print("before theta.T shape", theta.T.shape)
    # print("y.shape: ", y.shape)  # ('y.shape: ', (1, 2285))

    h = np.dot(theta.T, np.vstack((np.ones((1, x.shape[1])), x)))
    # print("h.shape: ", h.shape)
    correct = 0
    for i in range(y.shape[1]):
        if (y[0, i] == 1 and h[0, i] > 0) or (y[0, i] == 0 and h[0, i] < 0):
            correct += 1
    # print("=======================================")
    # print("=> ", float(correct) / y.shape[1])
    return float(correct) / y.shape[1]


def grad_descent(train_X, train_y, validation_X, validation_y, test_X, test_y,
                 f, df, x, y, theta0, alpha, _lambda, max_iter=10000):
    EPS=1e-5
    prev_t = theta0 - 10 * EPS
    theta = theta0.copy()
    iter = 0
    itr_idx = []
    train_performance, validation_performance, test_performance = [], [], []
    while np.linalg.norm(theta - prev_t) > EPS and iter < max_iter:
        prev_t = theta.copy()
        theta -= alpha * df(train_X, train_y, theta, _lambda)
        if iter % 10 == 0:
            print("----", iter, "----")
            itr_idx.append(iter)
            train_performance.append(performance(train_X, train_y, theta))
            validation_performance.append(performance(validation_X, validation_y, theta))
            test_performance.append(performance(test_X, test_y, theta))
        iter += 1

    return train_performance, validation_performance, test_performance, itr_idx, theta

def get_X(data, all_words, size):
    # print("============================")
    # print("how many words?: ", size)
    # print("how many headlines?: ", len(data))
    X = np.empty((size, 0))
    for line in data:
        each_x = np.zeros((size, 1))
        for i, word in enumerate(all_words):
            if word in line:
                each_x[i, 0] = 1
        X = np.hstack((X, each_x))
    # print("output X shape: ", X.shape)
    # print("============================")
    return X

def part4():
    real_data = read_data("./clean_real.txt")
    fake_data = read_data("./clean_fake.txt")

    real_train, real_valid, real_test = get_set(real_data)
    fake_train, fake_valid, fake_test = get_set(fake_data)
    print("dividing sets...")


    all_words = []
    for i in fake_train:
        all_words.extend(i)
    for i in real_train:
        all_words.extend(i)
    all_words = sorted(set(all_words), key=all_words.index)
    print("wordList length", len(all_words))  # 4086 words

    print("Getting X label and y labels...")

    size = len(all_words)

    train_X = np.concatenate((get_X(real_train, all_words, size), get_X(fake_train, all_words, size)), axis=1)
    train_y = np.concatenate((np.zeros((1, len(real_train))), np.ones((1, len(fake_train)))), axis=1)

    # 2285 = 1377 real headlines + 908 fake headlines
    # print("train_X shape:", train_X.shape)  # ('train_X shape:', (4806, 2285))
    # print("train_y shape:", train_y.shape)  # ('train_y shape:', (1, 2285))

    validation_X = np.concatenate((get_X(real_valid, all_words, size), get_X(fake_valid, all_words, size)), axis=1)
    validation_y = np.concatenate((np.zeros((1, len(real_valid))), np.ones((1, len(fake_valid)))), axis=1)

    # print("validation_X shape:", validation_X.shape)  # ('validation_X shape:', (4806, 490))
    # print("validation_y shape:", validation_y.shape)  # ('validation_y shape:', (1, 490))

    test_X = np.concatenate(((get_X(real_test, all_words, size), get_X(fake_test, all_words, size))), axis=1)
    test_y = np.concatenate((np.zeros((1, len(real_test))), np.ones((1, len(fake_test)))), axis=1)

    # print("test_X shape:", test_X.shape)  # ('test_X shape:', (4806, 491))
    # print("test_y shape:", test_y.shape)  # ('test_y shape:', (1, 491))

    np.random.seed(0)
    alpha = 0.0001
    _lambda = 0.1
    theta0 = np.random.normal(scale=alpha, size=(len(all_words)+1, 1))

    train_p, validation_p, test_p, itrs, t = grad_descent(train_X, train_y, validation_X, validation_y, test_X, test_y,
                 f, df, train_X, train_y, theta0, alpha, _lambda)

    print("train_p", train_p)

    fig = plt.figure()
    plt.title("Part4: Learning curve")
    plt.plot(itrs, train_p, label="Training")
    plt.plot(itrs, validation_p, label="Validation")
    plt.plot(itrs, test_p, label="Test")
    plt.ylabel("Performance")
    plt.xlabel("Iteration")
    plt.legend(loc="best")
    plt.savefig('part4_learning_curve.png')

    return t, all_words


#part4()

# ==============================================================================
#                        PART 5
# ==============================================================================




# ==============================================================================
#                        PART 6
# ==============================================================================

def part6a():

    t, all_words = part4()
    all_thetas = t.flatten().tolist()
    max_min = sorted(all_thetas, reverse=True)
    top10_positive, top10_negative = [], []

    for i in range(10):
        word = all_words[all_thetas.index(max_min[i])-1]
        top10_positive.append([word, max_min[i]])
    for i in np.arange(-1,-11,-1):
        word = all_words[all_thetas.index(max_min[i])-1]
        top10_negative.append([word, max_min[i]])

    print("Top10 positive theta(s) obtained from Logistic Regression with the words")
    for e in top10_positive:
        print("The %i th word is %s with theta: %07.7f"%(top10_positive.index(e)+1, e[0], e[1]))

    print("Top10 negative theta(s) obtained from Logistic Regression with the words")
    for e in top10_negative:
        print("The %i th word is %s with theta: %07.7f"%(top10_negative.index(e)+1, e[0], e[1]))


    return 0

# part6a()

def part6b():

    t, all_words = part4()
    all_thetas = t.flatten().tolist()
    max_min = sorted(all_thetas, reverse=True)
    top10_positive, top10_negative = [], []

    i, count = 0, 0
    while count < 10:
        word = all_words[all_thetas.index(max_min[i])-1]
        if word not in ENGLISH_STOP_WORDS:
            top10_positive.append([word, max_min[i]])
            count += 1
        i += 1

    i, count = -1, 0
    while count < 10:
        word = all_words[all_thetas.index(max_min[i])-1]
        if word not in ENGLISH_STOP_WORDS:
            top10_negative.append([word, max_min[i]])
            count += 1
        i -= 1

    print("Top10 positive theta(s) obtained from Logistic Regression without stopwords")
    for e in top10_positive:
        print("The %i th word is %s with theta: %07.7f"%(top10_positive.index(e)+1, e[0], e[1]))

    print("Top10 negative theta(s) obtained from Logistic Regression without stopwords")
    for e in top10_negative:
        print("The %i th word is %s with theta: %07.7f"%(top10_negative.index(e)+1, e[0], e[1]))


    return 0

#part6b()


# ==============================================================================
#                        PART 7
# ==============================================================================

def change_set(set, label):
    """
    change the form of the set to the form of decision tree's
    output set: array-like or sparse matrix, shape = [n_samples, n_features]
    output label: array-like, shape = [n_samples] or [n_samples, n_outputs]
    """
    set = np.array(set).T
    label = np.array(label).flatten()
    return set, label

def correctness_7(clf, set, label):
    correct = 0
    for i in range(len(clf.predict(set))):
        if label[i] == clf.predict(set)[i]:
            correct += 1
            # print correct
    return correct

def part7():
    real_data = read_data("./clean_real.txt")
    fake_data = read_data("./clean_fake.txt")

    f_train, f_valid, f_test = get_set(fake_data)
    r_train, r_valid, r_test = get_set(real_data)

    total_word = []
    for word in r_train:
        total_word.append(word)
    for word in f_train:
        total_word.append(word)

    total_word = flatten(total_word)
    total_word = sorted(set(total_word), key=total_word.index)

    size = len(total_word)

    train_x = np.concatenate((get_X(r_train, total_word, size), get_X(f_train, total_word, size)), axis=1)
    train_y = np.concatenate((np.zeros((1, len(r_train))), np.ones((1, len(f_train)))), axis=1)

    validation_x = np.concatenate((get_X(r_valid, total_word, size), get_X(f_valid, total_word, size)), axis=1)
    validation_y = np.concatenate((np.zeros((1, len(r_valid))), np.ones((1, len(f_valid)))), axis=1)

    test_x = np.concatenate((get_X(r_test, total_word, size), get_X(f_test, total_word, size)), axis=1)
    test_y = np.concatenate((np.zeros((1, len(r_test))), np.ones((1, len(f_test)))), axis=1)

    train_set, train_set_y = change_set(train_x, train_y)

    valid_set, valid_set_y = change_set(validation_x, validation_y)

    test_set, test_set_y = change_set(test_x, test_y)


    depth = np.arange(10, 210, 20)
    best_performance_va = 0
    best_performance_tr = 0
    best_performance_te = 0
    best_depth = 0
    perform_train = []
    perform_valid = []

    print "=============== part7a ================"

    for d in depth:
        print "d:", d
        clf = tree.DecisionTreeClassifier(max_depth = d, criterion = 'entropy')
        clf = clf.fit(train_set, train_set_y)
        perform_tr = float(correctness_7(clf, train_set, train_set_y))/len(train_set_y)
        perform_va = float(correctness_7(clf, valid_set, valid_set_y))/len(valid_set_y)
        perform_te = float(correctness_7(clf, valid_set, valid_set_y))/len(valid_set_y)
        print "training corr", perform_tr
        print "correctness", perform_va
        perform_train.append(perform_tr)
        perform_valid.append(perform_va)
        print "performace", perform_va
        print best_performance_va
        if best_performance_va < perform_va:
            best_depth = d
            best_performance_va = perform_va
            best_performance_tr = perform_tr
            best_performance_te = perform_te


    plt.figure()
    plt.plot(depth, perform_train, marker = "o", label = "Trainining set")
    plt.plot(depth, perform_valid, marker = "o", label = "Validation set")
    plt.title("Relationship between max_depth and Performance")
    plt.ylabel("Performance")
    plt.xlabel("depth")
    plt.legend(loc = "best")
    plt.savefig("part7a")

    print "best depth:", best_depth
    print "best performance training:", best_performance_tr
    print "best performance validation:", best_performance_va
    print "best performance test:", best_performance_te

    print "=============== part7b ================"

    clf = tree.DecisionTreeClassifier(max_depth = 110, criterion = 'entropy')
    clf = clf.fit(train_set, train_set_y)

    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=2,
                                    feature_names=total_word, rounded=True,
                                    filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("part7b","./", view=True)





# part7()
part2()
# part3()
