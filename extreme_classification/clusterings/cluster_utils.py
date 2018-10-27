import numpy as np


class CoOccurrenceDistance:

    def __init__(self, common_occurrences):
        self.common_occurrences = common_occurrences
        print(self.common_occurrences[0, 1])

    def __call__(self, cluster1, cluster2):
        # print("C1: ", cluster1)
        # print("C2: ", cluster2)
        assert ~any(i in cluster1 for i in cluster2), "Clusters are not disjoint"
        max_distance = -np.inf
        ids = None
        for i in range(len(cluster1)):
            for j in range(len(cluster2)):
                # print("i:", i, "j:", j)
                distance = -self.common_occurrences[cluster1[i], cluster2[j]]
                if distance > max_distance:
                    max_distance = distance
                    ids = (i, j)
        assert ids is not None
        return max_distance, ids
