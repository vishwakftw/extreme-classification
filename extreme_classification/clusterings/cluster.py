

class Cluster(object):

    def __init__(self, indexes):
        self.indexes = indexes
        self.indexes.sort()

    def merge(self, other_cluster):
        assert ~any(i in self.indexes for i in other_cluster.indexes), "Clusters are not disjoint"
        self.indexes = self.indexes + other_cluster.indexes
        self.indexes.sort()
        return self

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.indexes[idx]

    def __repr__(self):
        return self.indexes.__repr__()

    def __str__(self):
        return self.indexes.__str__()
