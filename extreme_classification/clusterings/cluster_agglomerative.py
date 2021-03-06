import sklearn
import sklearn.cluster


class CoOccurrenceAgglomerativeClustering(object):
    """
    Performs Agglomerative Clustering using negative of co-occurrence as the distance metric

    Accepts a matrix of classes, and performs clustering on the classes of the dataset

    Args:
        class_matrix : The matrix from which the class data is obtained
    """

    def __init__(self, class_matrix):
        self.class_matrix = class_matrix
        self.vector_length = self.class_matrix.get_shape()[1]
        self.num_data_points = self.class_matrix.get_shape()[0]
        self.distances = -(self.class_matrix.T * self.class_matrix)
        self.model = sklearn.cluster.AgglomerativeClustering(n_clusters=self.vector_length,
                                                             affinity='precomputed',
                                                             memory='/tmp/XC',
                                                             compute_full_tree=True,
                                                             linkage='complete')
        self.model.fit(self.distances.toarray())

    def get_clusters(self, num_clusters=None):
        """
        Performs clustering on the classes of the dataset and returns clusters

        Args:
            num_clusters : The number of clusters to create

        Returns:
            A 1D array with the ith element being the cluster index of the ith data point
        """
        if num_clusters is None:
            num_clusters = self.vector_length
        self.model.set_params(n_clusters=num_clusters)
        clusters = self.model.fit_predict(self.distances.toarray())
        return clusters

    def get_cluster_merge_indices(self):
        """
        Gets the classes in each cluster at each iteration of merging

        Returns:
            A 2D array, where the ith element contains class indices in the ith cluster id
        """
        ordering = [[i] for i in range(self.vector_length)]
        iterations = self.model.children_
        for i in range(self.vector_length - 1):
            ordering.append(ordering[iterations[i][0]] + ordering[iterations[i][1]])
        return ordering

    def get_merge_iterations(self):
        """
        Gets the order in which classes are merged during clustering

        Returns:
            A 1D array, with the ith element consisting of the two cluster ids that are merged in
            the ith step
        """
        return self.model.children_

    def get_ordering(self):
        """
        Gets an ordering of classes of the dataset based on clusters

        Returns:
            A 1D array that is a permutation of the class IDs
        """
        ordering = {i: [i] for i in range(self.vector_length)}
        iterations = self.model.children_
        c = len(ordering)
        for i in range(self.vector_length - 1):
            ordering[c] = ordering[iterations[i][0]] + ordering[iterations[i][1]]
            del ordering[iterations[i][0]]
            del ordering[iterations[i][1]]
            c += 1
        return ordering[c - 1]
        # ordering = [[i] for i in range(self.vector_length)]
        # iterations = self.model.children_
        # for i in range(self.vector_length - 1):
        #     ordering.append(ordering[iterations[i][0]] + ordering[iterations[i][1]])
        # return ordering[-1]

    def get_model(self):
        """
        Gets the sklearn.cluster.AgglomerativeClustering model used for clustering

        Returns:
            Instance of sklearn.cluster.AgglomerativeClustering used for clustering
        """
        return self.model

    def get_distance_matrix(self):
        """
        Gets the matrix of distances between any two data points based on negative
        co-occurrence

        Returns:
            A 2D matrix where the (ij)th element is the distance between the ith data point
            and the jth data point
        """
        return self.distances
