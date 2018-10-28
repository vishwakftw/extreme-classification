import os
import sklearn
import sklearn.cluster
import sys

sys.path.append('..')  # TODO: fix this
from loaders.loader_libsvm import LibSVMLoader


class CoOccurrenceAgglomerativeClustering(object):
    """
    Performs Agglomerative Clustering using negative of co-occurrence as the distance metric

    Accepts a data loader, and performs clustering on the classes of the dataset 
    """

    def __init__(self, loader):
        """
        @brief      Initialization

        @param      self    The object
        @param      loader  The data loader from which the class data is obtained

        """
        self.class_matrix = loader.get_classes()
        self.distances = -(self.class_matrix * self.class_matrix.T)
        self.model = sklearn.cluster.AgglomerativeClustering(affinity='precomputed', memory=os.path.join(
            '/', 'tmp', 'extreme_classification'), compute_full_tree=True, linkage='complete')

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

    def get_distance_matrix(self):
        """
        @brief      Gets the matrix of distances between any two data points based on negative 
                    co-occurrence

        @param      self  The object

        @return     A 2D matrix where the (ij)th element is the distance between the ith data point
                    and the jth data point
        """
        return self.distances

if __name__ == "__main__":
    print("Performs Agglomerative Clustering using negative of co-occurrence as the distance metric")
