import numpy as np
import pandas as pd
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

def write_results(filepath: str, data: list, columns: list):
    with open(filepath, 'w') as results_file:
        csv_writer = csv.DictWriter(results_file, fieldnames=columns)
        csv_writer.writeheader()
        for d in data:
            csv_writer.writerow({ column: value for value, column in zip(d, columns) })
        
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

    k_folds = [5, 8, 10]
    k_neighbors = [3, 5, 7]
    for kf in k_folds:
        for kn in k_neighbors:
            classifier = KNNClassifier(number_of_neighbors = kn)
            cross_validator = KFoldCrossValidator(number_of_folds= kf, seed=42, verbose=True)
            results = cross_validator.validate(classifier, data, labels)
            write_results(f'results/{kf}_{kn}.csv', results, ['k', 'Accuracy', 'F1-score'])
