from ..clusterings import CoOccurrenceAgglomerativeClustering

import numpy as np
import scipy.sparse as ssp


class HierarchicalXC(object):
    """
    Performs Hierarchical XC
    """

    def __init__(self):
        pass

    def train(self, feature_matrix, class_matrix, base_classifier, **kwargs):  # TODO add max_depth
        """
        Trains the tree of classifiers on a given dataset

        Args:
            feature_matrix : scipy csr matrix of input features
            class_matrix : scipy csr matrix of output class labels
            base_classifier : classifier to use for each tree node
            kwargs : parameters to pass to base_classifier
        """
        self.base_classifier = base_classifier
        self.classifier_params = kwargs
        self.feature_matrix, self.class_matrix = feature_matrix, class_matrix
        self.cluster_creator = CoOccurrenceAgglomerativeClustering(self.class_matrix)
        self.merge_indices = self.cluster_creator.get_cluster_merge_indices()
        self.merge_iterations = self.cluster_creator.get_merge_iterations()
        self.classifiers = [None] * len(self.merge_iterations)
        self.num_classes = self.class_matrix.shape[1]
        for merge in range(len(self.merge_iterations) - 1, -1, -1):
            class_indexes = []
            for i in range(2):
                class_i_list = self.merge_indices[self.merge_iterations[merge][i]]
                class_i_indexes = ssp.find(self.class_matrix[:, class_i_list] == 1)
                class_indexes.append(class_i_indexes[0])
            data_only_0 = np.setdiff1d(class_indexes[0], class_indexes[1], assume_unique=True)
            data_only_1 = np.setdiff1d(class_indexes[1], class_indexes[0], assume_unique=True)
            data_both = np.intersect1d(class_indexes[0], class_indexes[1], assume_unique=True)
            l_0 = data_only_0.shape[0]
            l_1 = data_only_1.shape[0]
            l_both = data_both.shape[0]
            train_X = ssp.vstack((self.feature_matrix[data_only_0], self.feature_matrix[
                data_only_1], self.feature_matrix[data_both]))
            train_y = np.zeros((l_0 + l_1 + l_both, 2))
            train_y[:l_0, 0] = 1
            train_y[l_0:l_1, 1] = 1
            train_y[l_1:] = 1
            classifier, is_true_classifier = self.train_single_classifier(train_X, train_y)
            self.classifiers[merge] = (classifier, is_true_classifier)

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
            return None, False
        clf = self.base_classifier(**self.classifier_params)
        clf.fit(train_X, train_y)
        return clf, True

    def predict(self, X):
        """
        Predicts classes given data

        Args:
            X : the dataset to predict on

        Returns:
            classes : the class predictions for each data point
        """
        self.test_X = X.toarray()
        start_id = len(self.merge_iterations) - 1
        self.classes = ssp.lil_matrix(np.zeros((len(self.test_X), self.num_classes)))
        ids = np.arange(len(self.test_X))
        self.traverse_classifiers(start_id, ids)
        return self.classes

    def traverse_classifiers(self, current_id, this_ids):
        """
        Traverses a node in the tree of classifers, and recursively calls itself to traverse child
        nodes

        Args:
            current_id : position of node in the list of classifiers representing the tree
            X : data handled by the node
            classes : class predictions for all the test data points
        """
        classifier, is_true_classifier = self.classifiers[current_id]
        if not is_true_classifier:
            preds = np.array([[1, 1]] * len(self.test_X[this_ids]))
        else:
            preds = classifier.predict(self.test_X[this_ids])
        ids = []
        classifier_ids = []
        for c in range(2):
            ids = this_ids[np.where(preds[:, c] == 1)[0]]
            if(len(ids)) == 0:
                continue
            classifier_ids = self.merge_iterations[current_id][c]
            if classifier_ids < self.num_classes:
                self.classes[ids, classifier_ids] = 1
            else:
                self.traverse_classifiers(classifier_ids - self.num_classes, ids)
