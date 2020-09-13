import numpy as np
import csv

from classification import KNNClassifier
from validation import KFoldCrossValidator

def read_dataset(filepath: str, label_column: str):
    data = []
    labels = []
    with open(filepath) as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        columns = next(csv_reader)
        label_column_index = columns.index(label_column)
        for row in csv_reader:
            data.append(list(map(lambda x: float(x), row[:label_column_index])))
            labels.append(row[label_column_index])
    return np.array(data), np.array(labels)

def normalize_values(data: np.array):
    data_transposed = np.transpose(data)
    for feature_data in data_transposed:
        min_value = np.min(feature_data)
        max_value = np.max(feature_data)
        feature_data -= min_value
        feature_data /= (max_value - min_value)
    return np.transpose(data_transposed)

if __name__ == '__main__':
    data, labels = read_dataset('data/diabetes.csv', 'Outcome')
    data = normalize_values(data)

    classifier = KNNClassifier(number_of_neighbors = 5)
    cross_validator = KFoldCrossValidator(number_of_folds= 8)
    cross_validator.validate(classifier, data, labels)

    # result = classifier.predict(data[1:], labels[1:], data[0])
    # print(result)