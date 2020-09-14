import numpy as np


class KFoldCrossValidator:
    def __init__(self, number_of_folds: int, seed: int = 42, verbose: bool = False):
        self.number_of_folds = number_of_folds
        self.seed = seed
        self.verbose = verbose

    def validate(self, classifier, data, labels, seed=42):
        folds_data, folds_labels = self._generate_folds(data, labels)
        results = []
        for k in range(self.number_of_folds):
            if (self.verbose): 
                print(f'Current testing fold: {k + 1}')
            test_data = folds_data[k]
            test_labels = folds_labels[k]
            train_data, train_labels = self._append_train_folds(folds_data, folds_labels, k)
            predicted_labels = np.array([])
            for test_data_element in test_data:
                predicted_labels = np.append(predicted_labels, classifier.predict(train_data, train_labels, test_data_element))
            results.append((k + 1, self._calculate_accuracy(test_labels, predicted_labels), self._calculate_f1_score(test_labels, predicted_labels)))
        return results

    def _append_train_folds(self, folds_data, folds_labels, current_test_fold: int):
        train_data = np.empty((0, folds_data[0].shape[1]))
        train_labels = np.array([])
        for i in range(self.number_of_folds):
            if i != current_test_fold:
                train_data = np.append(train_data, folds_data[i], axis=0)
                train_labels = np.append(train_labels, folds_labels[i])
        return train_data, train_labels

    def _generate_folds(self, data, labels):
        data_grouped_by_labels = self._group_data_by_class(data, labels)
        total_number_of_elements = len(data)
        number_of_elements_by_fold = np.floor(total_number_of_elements / self.number_of_folds)
        number_of_elements_to_sample_by_lable = {}

        for l in data_grouped_by_labels:
            elements_by_group = len(data_grouped_by_labels[l])
            group_proportion = elements_by_group / total_number_of_elements
            number_of_elements_to_sample = np.ceil(number_of_elements_by_fold * group_proportion)
            number_of_elements_to_sample_by_lable[l] = number_of_elements_to_sample
            
        np.random.seed(self.seed)
        folds_data = []
        folds_labels = []

        for n in range(self.number_of_folds):
            new_fold_data = np.empty((0, data.shape[1]))
            labels = np.array([])
            number_of_elements_sampled_by_label = {'0': 0, '1': 0}
            for l in number_of_elements_to_sample_by_lable:
                sampled_indexes = np.random.choice(
                    len(data_grouped_by_labels[l]), 
                    min(int(number_of_elements_to_sample_by_lable[l]), len(data_grouped_by_labels[l])), 
                    replace=False
                )

                number_of_elements_sampled_by_label[l] = min(int(number_of_elements_to_sample_by_lable[l]), len(data_grouped_by_labels[l]))

                for i in sampled_indexes:
                    new_fold_data = np.append(new_fold_data, [data_grouped_by_labels[l][i]], axis=0)
                    labels = np.append(labels, l)
                data_grouped_by_labels[l] = np.delete(data_grouped_by_labels[l], sampled_indexes, axis=0)
            folds_data.append(new_fold_data)
            folds_labels.append(labels)

            if (self.verbose):
                print(f'Fold {n + 1} generated')
                print('Percentage of negative values: ', number_of_elements_sampled_by_label['0'] / (number_of_elements_sampled_by_label['0'] + number_of_elements_sampled_by_label['1']))
                print('Percentage of positive values: ', number_of_elements_sampled_by_label['1'] / (number_of_elements_sampled_by_label['0'] + number_of_elements_sampled_by_label['1']))

        return folds_data, folds_labels

    def _group_data_by_class(self, data, labels):
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
        return round(number_of_correct_predictions / len(correct_labels), 3)

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
        return round(2 * (precision * recall) / (precision + recall), 3) 