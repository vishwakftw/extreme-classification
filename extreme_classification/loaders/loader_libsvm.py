import os
import torch
import itertools
import numpy as np
import torch.utils.data

from scipy.sparse import csr_matrix


def _parse_data_point(data_point):
    """
    Function to parse a row representing one input-output(s) pair in the LIBSVM format

    Args:
        data_point : row string

    Returns:
        (class_list, cols, vals) : class_list is the set of outputs for input,
                                   cols are the indices of values
                                   vals are the values taken at the indices
    """
    class_list = None
    elems = data_point.split(' ')
    num_nonzero_features = len(elems)

    if ':' not in elems[0]:
        class_list = np.array(list(map(int, elems[0].split(','))))
        num_nonzero_features -= 1
        elems = elems[1:]

    cols = np.empty(num_nonzero_features, dtype=int)
    vals = np.empty(num_nonzero_features, dtype=float)

    for i in range(num_nonzero_features):
        idx, val = elems[i].split(':')
        cols[i] = int(idx)
        vals[i] = float(val)
    return class_list, cols, vals


class LibSVMLoader(torch.utils.data.Dataset):
    """
    Class for a dataset in the LibSVM format.
    """

    def __init__(self, file_path=None, dataset_info=None, feature_matrix=None, class_matrix=None):
        """
        Initializes the loader. Either file_path and dataset_info, or feature_matrix and
        class_matrix, must be provided. If both are provided, data is loaded from the file.

        Args:
            file_path : Path to the file containing the dataset. The file should only consists of
                        rows of datum in the LibSVM format.
            dataset_info : Dictionary consisting of three fields `num_data_points`, `input_dims`
                           and `output_dims`.
            feature_matrix : Precomputed feature_matrix.
            class_matrix : Precomputed class_matrix.
        """
        assert (file_path is not None and dataset_info is not None) or (
            feature_matrix is not None and class_matrix is not None), \
            "Either file path, or feature and class matrices must be specified"
        if file_path is not None:
            assert os.path.isfile(file_path), file_path + " does not exist!"
            self.num_data_points = dataset_info['num_data_points']
            self.input_dims = dataset_info['input_dims']
            self.output_dims = dataset_info['output_dims']

            with open(file_path, 'r') as f:
                data = f.readlines()
                assert self.num_data_points == len(data), "Mismatch in number of data points"
                class_matrix = []
                data_rows_matrix = []
                class_rows_matrix = []
                cols_matrix = []
                data_matrix = []

                for i in range(self.num_data_points):
                    class_list, cols, vals = _parse_data_point(data[i])
                    class_matrix.append(class_list)
                    cols_matrix.append(cols)
                    data_matrix.append(vals)
                    data_rows_matrix.append(np.full(len(cols_matrix[i]), i))
                    class_rows_matrix.append(np.full(len(class_matrix[i]), i))

                class_matrix = list(itertools.chain.from_iterable(class_matrix))
                data_rows_matrix = list(
                    itertools.chain.from_iterable(data_rows_matrix))
                class_rows_matrix = list(
                    itertools.chain.from_iterable(class_rows_matrix))
                cols_matrix = list(itertools.chain.from_iterable(cols_matrix))
                data_matrix = list(itertools.chain.from_iterable(data_matrix))

                assert len(data_matrix) == len(data_rows_matrix) and len(
                    data_matrix) == len(cols_matrix)
                assert len(class_rows_matrix) == len(class_matrix)

                self.features = csr_matrix((data_matrix, (data_rows_matrix, cols_matrix)),
                                           shape=(self.num_data_points, self.input_dims))
                self.classes = csr_matrix((np.ones(len(class_rows_matrix)),
                                           (class_rows_matrix, class_matrix)),
                                          shape=(self.num_data_points, self.output_dims))
        else:
            assert feature_matrix.get_shape()[0] == class_matrix.get_shape()[
                0], "Mismatch in number of features and classes"
            self.num_data_points = feature_matrix.get_shape()[0]
            self.input_dims = feature_matrix.get_shape()[1]
            self.output_dims = feature_matrix.get_shape()[1]
            self.features = feature_matrix
            self.classes = class_matrix

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx].todense().reshape(-1)),
                torch.from_numpy(self.classes[idx].todense().reshape(-1)))

    def __repr__(self):
        fmt_str = 'Sparse dataset of size ({0} x {1}), ({0} x {2})'.format(
            len(self), self.input_dims, self.output_dims)
        fmt_str += ' in LIBSVM format'
        return fmt_str

    def train_test_split(self, test_fraction=0.2, random_seed=42):
        """
        Function to split the data randomly into train and test splits in a specified ratio.

        Args:
            test_fraction : Fraction of elements to keep in the test set.
            random_seed : Random seed for shuffling data before splitting.

        Returns:
            (train_loader, test_loader) : Loaders containing the train and test splits respectively

        """
        assert test_fraction >= 0.0 and test_fraction <= 1.0, "Test set fraction must lie in [0,1]"
        np.random.seed(random_seed)
        permutation = np.random.shuffle(np.arange(self.num_data_points))
        split_index = int(self.num_data_points * (1 - test_fraction))
        train_loader = LibSVMLoader(feature_matrix=self.features[
                                    :split_index], class_matrix=self.classes[:split_index])
        test_loader = LibSVMLoader(feature_matrix=self.features[
                                   split_index:], class_matrix=self.classes[split_index:])
        return train_loader, test_loader

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return (self.features, self.classes)

    def get_features(self):
        """
        Function to get the entire set of features
        """
        return self.features

    def get_classes(self):
        """
        Function to get the entire set of classes
        """
        return self.classes

    def num_classes(self):
        """
        Function to get number of classes in the dataset 
        """
        return self.output_dims
