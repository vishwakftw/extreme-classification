import numpy as np
from ..clusterings import CoOccurrenceAgglomerativeClustering
import sys
import scipy
import sklearn


class DummyClassifier(object):

    def predict(self, X):
        return np.array([[1, 1]] * len(X))


class HierarchicalXC(object):
    """
    Performs Hierarchical XC
    """

    def __init__(self):
        pass

    def train(self, loader, base_classifier, **kwargs):  # TODO add max_depth
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
        self.merge_indices = self.cluster_creator.get_cluster_merge_indices()
        self.merge_iterations = self.cluster_creator.get_merge_iterations()
        self.classifiers = [None] * len(self.merge_iterations)
        self.num_classes = self.loader.num_classes()
        for merge in range(len(self.merge_iterations) - 1, -1, -1):
            class_indexes = []
            for i in range(2):
                class_i_list = self.merge_indices[self.merge_iterations[merge][i]]
                class_i_indexes = scipy.sparse.find(self.class_matrix[:, class_i_list] == 1)
                class_indexes.append(class_i_indexes[0])
            data_only_0 = np.setdiff1d(class_indexes[0], class_indexes[1], assume_unique=True)
            data_only_1 = np.setdiff1d(class_indexes[1], class_indexes[0], assume_unique=True)
            data_both = np.intersect1d(class_indexes[0], class_indexes[1], assume_unique=True)
            l_0 = data_only_0.shape[0]
            l_1 = data_only_1.shape[0]
            l_both = data_both.shape[0]
            train_X = scipy.sparse.vstack((self.feature_matrix[data_only_0], self.feature_matrix[
                                          data_only_1], self.feature_matrix[data_both]))
            train_y = np.zeros((l_0 + l_1 + l_both, 2))
            train_y[:l_0, 0] = 1
            train_y[l_0:l_1, 1] = 1
            train_y[l_1:] = 1
            classifier = self.train_single_classifier(train_X, train_y)
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
        assert len(train_X) == len(train_y), "Size mismatch in data points and labels"
        if len(train_y) == 0:
            return DummyClassifier()
        # self.base_classifier(self.classifier_params)
        clf = sklearn.multiclass.OneVsRestClassifier(
            sklearn.svm.SVC(gamma='scale'))  # TODO: Fix this
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
        X = X.toarray()
        start_id = len(self.merge_iterations) - 1
        classes = scipy.sparse.lil_matrix(np.zeros((len(X), self.num_classes)))
        self.traverse_classifiers(start_id, X, classes)
        return classes

    def traverse_classifiers(self, current_id, X, classes):
        """ 
        Traverses a node in the tree of classifers, and recursively calls itself to traverse child
        nodes

        Args:
            current_id : position of node in the list of classifiers representing the tree
            X : data handled by the node
            classes : class predictions for all the test data points
        """
        classifier = self.classifiers[current_id]
        preds = classifier.predict(X)
        id_0 = np.where(preds[:, 0] == 1)
        id_1 = np.where(preds[:, 1] == 1)
        X_0 = X[id_0]
        X_1 = X[id_1]
        classifier_0_id = self.merge_iterations[current_id][0]
        classifier_1_id = self.merge_iterations[current_id][1]
        if classifier_0_id < self.num_classes:
            classes[id_0, classifier_0_id] = 1
        else:
            self.traverse_classifiers(classifier_0_id - self.num_classes, X, classes)
        if classifier_1_id < self.num_classes:
            classes[id_1, classifier_1_id] = 1
        else:
            self.traverse_classifiers(classifier_1_id - self.num_classes, X, classes)
