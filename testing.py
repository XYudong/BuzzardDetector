from sklearn import svm
import pickle


def load(filename):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        data.append(list(map(float, line.split(' '))))

    print(str(len(data)) + ' * ' + str(len(data[0])))
    f.close()
    return data


method = 3
if method == 1:
    feature = 'SIFT'
elif method == 2:
    feature = 'SURF'
elif method == 3:
    feature = 'ORB'

model_path = 'pre-trained_models/'
filename = model_path + feature + '_model01.sav'
clf = pickle.load(open(filename, 'rb'), encoding='latin1')


dir_0 = 'output/'
dir_1 = 'output/'
pos_data = load(dir_0 + '/test_' + feature + '_positive01_0.txt')
pos_data.extend(load(dir_1 + '/test_' + feature + '_positive01_1.txt'))
neg_data = load(dir_0 + '/test_' + feature + '_negative01_0.txt')
neg_data.extend(load(dir_1 + '/test_' + feature + '_negative01_1.txt'))
data_test = pos_data + neg_data
label_test = [1]*len(pos_data) + [0]*len(neg_data)
output_test = []


output = clf.predict(data_test)
cnt = 0
for i in range(len(output)):
    if output[i] == label_test[i]:
        cnt += 1

print(cnt/float(len(output)))
