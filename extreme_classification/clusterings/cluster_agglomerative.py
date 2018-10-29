import os
import sklearn
import sklearn.cluster


class CoOccurrenceAgglomerativeClustering(object):
    """
    Performs Agglomerative Clustering using negative of co-occurrence as the distance metric

    Accepts a matrix of classes, and performs clustering on the classes of the dataset
    """

    def __init__(self, class_matrix):
        """
        @brief      Initialization

        @param      self    The object
        @param      class_matrix  The matrix from which the class data is obtained

        """
        self.class_matrix = class_matrix
        self.num_data_points = self.class_matrix.get_shape()[1]
        self.vector_length = self.class_matrix.get_shape()[0]
        self.distances = -(self.class_matrix.T * self.class_matrix)
        self.model = sklearn.cluster.AgglomerativeClustering(n_clusters=self.num_data_points,
                                                             affinity='precomputed',
                                                             memory=os.path.join(
                                                                 '/', 'tmp', 'extreme_classification'),
                                                             compute_full_tree=True, linkage='complete')
        self.model.fit(self.distances.toarray())

    def get_clusters(self, num_clusters):
        """
        @brief      Performs clustering on the classes of the dataset and returns clusters

        @param      self          The object
        @param      num_clusters  The number of clusters to create

        @return     A 1D array with the ith element being the cluster index of the ith data point
        """
        self.model.set_params(n_clusters=num_clusters)
        clusters = self.model.fit_predict(self.distances.toarray())
        return clusters

    def get_ordering(self):
        """
        @brief      Gets an ordering of classes of the dataset based on clusters

        @param      self  The object

        @return     A 1D array that is a permutation of the class IDs
        """
        ordering = [[i] for i in range(self.num_data_points)]
        iterations = self.model.children_
        for i in range(self.num_data_points - 1):
            ordering.append(ordering[iterations[i][0]] + ordering[iterations[i][1]])
        return ordering[-1]

    def get_model(self):
        """
        @brief      Gets the sklearn.cluster.AgglomerativeClustering model used for clustering

        @param      self  The object

        @return     The model
        """
        return self.model

    def get_distance_matrix(self):
        """
        @brief      Gets the matrix of distances between any two data points based on negative
                    co-occurrence

        @param      self  The object

        @return     A 2D matrix where the (ij)th element is the distance between the ith data point
                    and the jth data point
        """
        return self.distances
