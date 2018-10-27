import os
import torch
import itertools
import numpy as np
import torch
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

    def __init__(self, file_path):
        assert os.path.isfile(file_path), file_path + " does not exist!"

        with open(file_path, 'r') as f:
            data = f.readlines()
            self.num_data_points, self.input_dims, self.output_dims = list(
                map(int, data[0].split(' ')))
            assert self.num_data_points == len(
                data) - 1, "Mismatch in number of data points"
            class_matrix = []
            data_rows_matrix = []
            class_rows_matrix = []
            cols_matrix = []
            data_matrix = []

            for i in range(self.num_data_points):
                class_list, cols, vals = _parse_data_point(data[i + 1])
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

            self.features = csr_matrix((data_matrix, (data_rows_matrix, cols_matrix)), shape=(
                self.num_data_points, self.input_dims))
            self.classes = csr_matrix((np.ones(len(class_rows_matrix)), (class_rows_matrix, class_matrix)), shape=(
                self.num_data_points, self.output_dims))

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx].todense()),
                torch.from_numpy(self.classes[idx].todense()))

    def __repr__(self):
        fmt_str = 'Sparse dataset of size ({0} x {1}), ({0} x {2})'.format(
            len(self), self.input_dims, self.output_dims)
        fmt_str += ' in LIBSVM format'
        return fmt_str

    def get_data(self):
        return (self.features, self.classes)

    def get_features(self):
        return self.features

    def get_classes(self):
        return self.classes

    def num_classes(self):
        return self.output_dims
