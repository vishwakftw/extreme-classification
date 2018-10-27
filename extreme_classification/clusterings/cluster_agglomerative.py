import copy
import numpy as np
import os
import sys

from cluster import Cluster
from cluster_utils import CoOccurrenceDistance

sys.append('..')  # TODO: fix this
from ..loaders.loader_libsvm import LibSVMLoader


class AgglomerativeClustering(object):

    def __init__(self, loader):
        self.class_matrix = loader.get_classes()
        self.num_data_points = len(loader)
        self.num_classes = loader.num_classes()
        self.common_classes = self.class_matrix * self.class_matrix.T

    def get_hierarchy(self, distance_metric):
        cluster_hierarchy = [[Cluster([i]) for i in range(self.num_data_points)]]
        level = 0
        while(True):
            clusters_this_level = cluster_hierarchy[level]
            min_distance = np.inf
            clusters_to_merge = None
            print("level:", level)
            for i in range(len(clusters_this_level)):
                for j in range(i + 1, len(clusters_this_level)):
                    print(i, j)
                    distance, _ = distance_metric(clusters_this_level[i], clusters_this_level[j])
                    if distance < min_distance:
                        min_distance = distance
                        clusters_to_merge = (i, j)
            assert clusters_to_merge is not None
            clusters_next_level = copy.deepcopy(clusters_this_level)
            clusters_next_level[clusters_to_merge[0]].merge(
                clusters_next_level[clusters_to_merge[1]])

            clusters_next_level.remove(clusters_next_level[clusters_to_merge[1]])
            cluster_hierarchy.append(clusters_next_level)
            if len(clusters_next_level) == 1:
                break
            level += 1
        self.cluster_hierarchy = cluster_hierarchy
        return cluster_hierarchy

    def get_common_classes_matrix(self):
        return self.common_classes

if __name__ == "__main__":
    l = LibSVMLoader('../../data/Bibtex/Bibtex_data.txt')
    a = AgglomerativeClustering(l)
    print(a.get_hierarchy(CoOccurrenceDistance(a.get_common_classes_matrix())))
