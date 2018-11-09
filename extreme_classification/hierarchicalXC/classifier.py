import numpy as np


class HierarchicalXC(object):

    def __init__(self):
        pass

    def train(self, loader, base_classifier, max_depth=None):
        self.loader = loader
        self.base_classifier = base_classifier
        self.max_depth = max_depth
        feature_matrix, class_matrix = loader.get_data()
        cluster_creator = CoOccurrenceAgglomerativeClustering(class_matrix)
        clusters = cluster_creator.get_clusters()
