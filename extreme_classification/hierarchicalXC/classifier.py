import numpy as np
from ..clusterings import CoOccurrenceAgglomerativeClustering
import sys


class HierarchicalXC(object):
	"""
	Performs Hierarchical XC
	"""

    def __init__(self):
        pass

    def train(self, loader, base_classifier, **kwargs): # TODO add max_depth
    	""" 
		Trains the tree of classifiers on a given dataset

		Args:
			loader : loader with training data
			base_classifier : classifier to use for each tree node
			kwargs : parameters to pass to base_classifier
    	"""
        self.loader = loader
        self.base_classifier = base_classifier
        self.classifier_params = kwargs
        self.feature_matrix, self.class_matrix = loader.get_data()
        self.cluster_creator = CoOccurrenceAgglomerativeClustering(self.class_matrix)
        merge_indices = self.cluster_creator.get_cluster_merge_indices()
        merge_iterations = self.cluster_creator.get_merge_iterations()
        self.classifiers = [None] * len(merge_iterations)
        for merge in range(len(merge_iterations) - 1, -1, -1):
        	class_indexes = []
            for i in range(2):
                class_i_list = merge_indices[merge_iterations[merge][0]]
                class_i_indexes = scipy.sparse.find(self.class_matrix[:, class_i_list] == 1)
                class_indexes.append(class_i_indexes[0])
            data_only_0 = np.setdiff1d(class_indexes[0],class_indexes[1],assume_unique=True)
            data_only_1 = np.setdiff1d(class_indexes[1],class_indexes[0],assume_unique=True)
            data_both = np.intersect1d(class_indexes[0],class_indexes[1],assume_unique=True)
            l_0 = data_only_0.get_shape()[0]
            l_1 = data_only_1.get_shape()[0]
            l_both = data_both.get_shape()[0]
            train_X = scipy.sparse.vstack((data_only_0, data_only_1, data_both))
            train_y = np.zeros((l_0 + l_1 + l_both, 2))
            train_y[:l_0,0] = 1
            train_y[l_0:l_1,1] = 1
            train_y[l_1:] = 1
            classifier = train_single_classifier(train_X, train_y)
            self.classifiers[merge] = classifier

    def train_single_classifier(self, train_X, train_y):
    	""" 
		Trains a single classifier (a single tree node) on binary data

		Args:
			train_X : training data points 
			train_y : training class labels (0 or 1)

		Returns:
			clf : a trained classifier
    	"""
        train_X = train_X.toarray()
        clf = self.base_classifier(self.classifier_params)
        clf.fit(train_X, train_y)
        return clf

    def predict(self, X):
    	""" 
		Predicts classes given data 

		Args:
			X : the dataset to predict on

		Returns:
			classes : the class predictions for each data point
    	"""
        start_id = len(merge_iterations) - 1
        classes = [None] * len(X)
        traverse_classifiers(start_id, X, classes)
        return classes

    def traverse_classifiers(self, current_id, X, classes):
        classifier = self.classifiers[current_id]
