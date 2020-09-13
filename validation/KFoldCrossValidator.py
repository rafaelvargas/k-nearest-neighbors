import numpy as np


class KFoldCrossValidator:
    def __init__(self, number_of_folds):
        self.number_of_folds = number_of_folds

    def validate(self, classifier, data, labels):
        folds_data, folds_labels = self._sample_data_in_folds(data, labels)
        for i in range(len(folds_data)):
            print(f'Current fold: {i}')
            test_data = folds_data[i]
            test_labels = folds_labels[i]
            train_data = np.empty((0, 8))
            train_labels = np.array([])
            for j in range(len(folds_data)):
                if j != i:
                    train_data = np.append(train_data, folds_data[j], axis=0)
                    train_labels = np.append(train_labels, folds_labels[j])
            results = np.array([])
            for test_sample_data in test_data:
                results = np.append(results, classifier.predict(train_data, train_labels, test_sample_data))
            print(self._calculate_accuracy(test_labels, results))
            print(self._calculate_f1_score(test_labels, results))
    
    def _sample_data_in_folds(self, data, labels):
        data_grouped_by_label = self._group_by_labels(data, labels)
        number_of_elements = len(data)
        number_of_elements_by_fold = number_of_elements / self.number_of_folds # TODO: Should always be natural numbers
        number_of_elements_to_sample_by_lable = {}

        for l in data_grouped_by_label:
            elements_by_group = len(data_grouped_by_label[l])
            group_proportion = elements_by_group / number_of_elements
            number_of_elements_to_sample = number_of_elements_by_fold * group_proportion
            number_of_elements_to_sample_by_lable[l] = number_of_elements_to_sample

        folds_data = []
        folds_labels = []
        np.random.seed(42)
        for _ in range(self.number_of_folds):
            data = np.empty((0, 8)) # TODO: Should set the number of columns dinamically
            labels = np.array([])
            for l in number_of_elements_to_sample_by_lable:
                sampled_indexes = np.random.choice(
                    len(data_grouped_by_label[l]), 
                    int(number_of_elements_to_sample_by_lable[l]), 
                    replace=False
                )

                for i in sampled_indexes:
                    data = np.append(data, [data_grouped_by_label[l][i]], axis=0)
                    labels = np.append(labels, l)
                data_grouped_by_label[l] = np.delete(data_grouped_by_label[l], sampled_indexes, axis=0)
            folds_data.append(data)
            folds_labels.append(labels)
        return folds_data, folds_labels


    def _group_by_labels(self, data, labels):
        groups = {} 
        for d, l in zip(data, labels):
            try:
                groups[l] = np.vstack((groups[l], d))
            except KeyError:
                groups[l] = np.array(d)
        return groups

    def _calculate_accuracy(self, correct_labels: np.array, predicted_labels: np.array):
        number_of_correct_predictions = 0
        for correct_label, predicted_label in zip(correct_labels, predicted_labels):
            if (predicted_label == correct_label):
                number_of_correct_predictions += 1
        return number_of_correct_predictions / len(correct_labels)

    def _calculate_f1_score(self, correct_labels: np.array, predicted_labels: np.array):
        number_of_true_positives = 0
        number_of_false_positives = 0
        number_of_true_negatives = 0
        number_of_false_negatives = 0
        for correct_label, predicted_label in zip(correct_labels, predicted_labels):
            if (predicted_label == '1'):
                if (correct_label == predicted_label): 
                    number_of_true_positives += 1
                else:
                    number_of_false_positives += 1
            else:
                if (correct_label == predicted_label): 
                    number_of_true_negatives += 1
                else:
                    number_of_false_negatives += 1
        
        precision = number_of_true_positives / (number_of_true_positives + number_of_false_positives)
        recall = number_of_true_positives / (number_of_true_positives + number_of_false_negatives)
        return 2 * (precision * recall) / (precision + recall)