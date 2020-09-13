import numpy as np

class KNNClassifier:
    def __init__(self, number_of_neighbors: int):
        if (number_of_neighbors <= 0):
            raise ValueError('Invalid number of neighbors')
        self.number_of_neighbors = number_of_neighbors

    def predict(self, train_data: np.array, train_labels: np.array, test_data: np.array):
        self._check_train_dataset(train_data, train_labels)
        train_data = self._normalize_values(train_data)
        nearest_neighbors_distances = []
        nearest_neighbors_labels = []

        for data, label in zip(train_data, train_labels):
            distance = self._compute_euclidian_distance(data, test_data)
            if (len(nearest_neighbors_distances) < self.number_of_neighbors):
                nearest_neighbors_distances.append(distance)
                nearest_neighbors_labels.append(label)
            else:
                index = np.argmax(nearest_neighbors_distances)
                if nearest_neighbors_distances[index] > distance:
                    nearest_neighbors_distances[index] = distance
                    nearest_neighbors_labels[index] = label
        # print(nearest_neighbors_distances, nearest_neighbors_labels)
        return self._compute_mode(nearest_neighbors_labels)

    def _check_train_dataset(self, train_data: np.array, train_labels: np.array):
        if (len(train_data) < self.number_of_neighbors):
            raise ValueError('Invalid train dataset size')
        if (len(train_data) != len(train_labels)):
            print(len(train_data), len(train_labels))
            raise ValueError('The number of labels should be equal to the number of data elements')

    def _normalize_values(self, train_data: np.array):
        data_transposed = np.transpose(train_data)
        for feature_data in data_transposed:
            min_value = np.min(feature_data)
            max_value = np.max(feature_data)
            feature_data -= min_value
            feature_data /= (max_value - min_value)
        return np.transpose(data_transposed)

    def _compute_euclidian_distance(self, a: np.array, b: np.array):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    def _compute_mode(self, nearest_neighbors_labels: list):
        counting = {}
        for n in nearest_neighbors_labels:
            try:
                counting[n] += 1
            except KeyError as e:
                counting[n] = 1
        return max(counting.keys(), key=(lambda key: counting[key]))