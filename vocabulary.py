from descriptors_extraction import *
from sklearn.cluster import KMeans


def build_voc(desIn, n_clu):
    # rescale each feature dimension
    whitened = whiten(desIn)

    # do k-means to form a vocabulary
    kmeans = KMeans(n_clusters=n_clu, init='k-means++', random_state=66).fit(whitened)
    voc_out = kmeans.cluster_centers_  # the vocabulary

    return voc_out


fea_type = "SURF"
dataset = "train"
k = 50

# get a list of descriptors
positive_path = "data/" + dataset + "/positive/0/"
descriptors_wh_tr_pos = get_all_des(fea_type, positive_path)   # np array of whitened descriptors
negative_path = "data/" + dataset + "/negative/0/"
descriptors_wh_tr_neg = get_all_des(fea_type, negative_path)

descriptors_wh_tr = np.vstack((descriptors_wh_tr_pos, descriptors_wh_tr_neg))

print(descriptors_wh_tr.shape)
print('using ' + str(len(descriptors_wh_tr)) + ' descriptors as our corpus.\n')

# build vocabulary
print('building the vocabulary\n')

voc = build_voc(descriptors_wh_tr, k)
print('shape of vocabulary: ' + str(voc.shape) + '\n')

np.savetxt('./voc_output/myVoc_' + fea_type + '_0.txt', voc)