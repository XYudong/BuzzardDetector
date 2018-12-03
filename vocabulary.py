from descriptors_extraction import *
from sklearn.cluster import KMeans


def build_voc(desIn, n_clu):
    # rescale each feature dimension
    whitened = whiten(desIn)

    # do k-means to form a vocabulary
    kmeans = KMeans(n_clusters=n_clu, init='k-means++', random_state=66).fit(whitened)
    voc_out = kmeans.cluster_centers_  # the vocabulary

    return voc_out


def collect_descriptors(pos_path, neg_path):
    # get a list of descriptors
    pos = get_all_des(fea_type, pos_path)   # np array of whitened descriptors
    neg = get_all_des(fea_type, neg_path)
    out = np.vstack((pos, neg))
    return out


fea_type = "SIFT"
dataset = "train"

if fea_type == 'ORB':
    k = 30
else:
    k = 50

# get a list of descriptors
positive_path = "data/" + dataset + "/positive/0/"
negative_path = "data/" + dataset + "/negative/0/"
descriptors_0 = collect_descriptors(positive_path, negative_path)   # np array of whitened descriptors

positive_path = "data/" + dataset + "/positive/1/"
negative_path = "data/" + dataset + "/negative/1/"
descriptors_1 = collect_descriptors(positive_path, negative_path)

descriptors_tr = np.vstack((descriptors_0, descriptors_1))

print(descriptors_tr.shape)
print('using ' + str(len(descriptors_tr)) + ' descriptors as our corpus.\n')

# build vocabulary
print('building the vocabulary\n')

voc = build_voc(descriptors_tr, k)
print('shape of vocabulary: ' + str(voc.shape) + '\n')

np.savetxt('./voc_output/myVoc_' + fea_type + '_01.txt', voc)