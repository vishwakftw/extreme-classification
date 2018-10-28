import copy
import numpy as np
import os
import sklearn
import sklearn.cluster
import sys

from cluster import Cluster
from cluster_utils import CoOccurrenceDistance

sys.path.append('..')  # TODO: fix this
from loaders.loader_libsvm import LibSVMLoader


class CoOccurrenceAgglomerativeClustering(object):

    def __init__(self, loader):
        self.class_matrix = loader.get_classes()
        self.num_data_points = len(loader)
        self.num_classes = loader.num_classes()
        self.distances = self.class_matrix * self.class_matrix.T
        self.model = sklearn.cluster.AgglomerativeClustering(affinity='precomputed', memory=os.path.join(
            '/', 'tmp', 'extreme_classification'), compute_full_tree=True, linkage='complete')

    def get_clusters(self, num_clusters):
        self.model.set_params(n_clusters=num_clusters)
        clusters = self.model.fit_predict(self.distances.toarray())
        return clusters

    def get_distance_matrix(self):
        return self.distances

if __name__ == "__main__":
    print("Performs Agglomerative Clustering using negative of co-occurrence as the distance metric")
