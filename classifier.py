from sklearn.svm import SVC
from sklearn import svm
import numpy as np


#from sklearn import cross_validation, svm, metrics
#import matplotlib.pyplot as plt
#import pandas as pd


def load(filename):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        data.append(list(map(float,line.split(' '))))

    print(str(len(data)) + ' * ' + str(len(data[0])))
    f.close()
    return data


method = 2
if method == 1:
    feature = 'SIFT'
elif method == 2:
    feature = 'SURF'
elif method ==3:
    feature = 'ORB'

dir_0 = './data_subTask1/' + feature
dir_1 = './1/'
pos_data = load(dir_0 + '/train_' + feature + '_positive_0.txt')
pos_data.extend(load(dir_1 + '/train_' + feature + '_positive_1.txt'))
neg_data = load(dir_0 + '/train_' + feature + '_negative_0.txt')
neg_data.extend(load(dir_1 + '/train_' + feature + '_negative_1.txt'))

data_train = pos_data + neg_data
label_train = [1]*len(pos_data) + [0]*len(neg_data)
clf = svm.LinearSVC()
clf.fit(data_train, label_train)

pos_data = load(dir_0 + '/test_' + feature + '_positive_0.txt')
pos_data.extend(load(dir_1 + '/test_' + feature + '_positive_1.txt'))
neg_data = load(dir_0 + '/test_' + feature + '_negative_0.txt')
neg_data.extend(load(dir_1 + '/test_' + feature + '_negative_1.txt'))
data_test = pos_data + neg_data
label_test = [1]*len(pos_data) + [0]*len(neg_data)
output_test = []
#cnt = neg_data = load(dir_0 + '/train_' + feature + '_negative_0.txt')

output = clf.predict(data_test)
cnt = 0
for i in range(len(output)):
    if output[i] == label_test[i]:
        cnt += 1

print(cnt/float(len(output)))
