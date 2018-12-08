"""Credit from zhyma"""

# from sklearn.svm import SVC
from sklearn import svm
# import numpy as np
import pickle


def load(filename):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        data.append(list(map(float, line.split(' '))))

    print(str(len(data)) + ' * ' + str(len(data[0])))
    f.close()
    return data


method = 1
if method == 1:
    feature = 'SIFT'
elif method == 2:
    feature = 'SURF'
elif method == 3:
    feature = 'ORB'


dir_0 = 'output/'
dir_1 = 'output/'
pos_data = load(dir_0 + '/train_' + feature + '_positive01_0.txt')
pos_data.extend(load(dir_1 + '/train_' + feature + '_positive01_1.txt'))
neg_data = load(dir_0 + '/train_' + feature + '_negative01_0.txt')
neg_data.extend(load(dir_1 + '/train_' + feature + '_negative01_1.txt'))

data_train = pos_data + neg_data
label_train = [1]*len(pos_data) + [0]*len(neg_data)
clf = svm.LinearSVC()
clf.fit(data_train, label_train)

filename = feature + '_model01.sav'
pickle.dump(clf, open("pre-trained_models/" + filename, 'wb'))
