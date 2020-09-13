import numpy as np


class KFoldCrossValidator:
    def __init__(self, number_of_folds: int, seed: int=42):
        self.number_of_folds = number_of_folds
        self.seed = seed

    def validate(self, classifier, data, labels, seed=42):
        folds_data, folds_labels = self._sample_data_in_folds(data, labels)
        results = []
        for i in range(len(folds_data)):
            print(f'Current fold: {i + 1}')
            test_data = folds_data[i]
            test_labels = folds_labels[i]
            train_data = np.empty((0, 8))
            train_labels = np.array([])
            for j in range(len(folds_data)):
                if j != i:
                    train_data = np.append(train_data, folds_data[j], axis=0)
                    train_labels = np.append(train_labels, folds_labels[j])
            predicted_labels = np.array([])
            for test_sample_data in test_data:
                predicted_labels = np.append(predicted_labels, classifier.predict(train_data, train_labels, test_sample_data))
            results.append((i+1, self._calculate_accuracy(test_labels, predicted_labels), self._calculate_f1_score(test_labels, predicted_labels)))
        return results

    def _sample_data_in_folds(self, data, labels):
        data_grouped_by_label = self._group_by_labels(data, labels)
        number_of_elements = len(data)
        number_of_elements_by_fold = np.floor(number_of_elements / self.number_of_folds) # TODO: Should always be natural numbers
        number_of_elements_to_sample_by_lable = {}

        for l in data_grouped_by_label:
            elements_by_group = len(data_grouped_by_label[l])
            group_proportion = elements_by_group / number_of_elements
            number_of_elements_to_sample = np.ceil(number_of_elements_by_fold * group_proportion)
            number_of_elements_to_sample_by_lable[l] = number_of_elements_to_sample
            
        folds_data = []
        folds_labels = []
        np.random.seed(self.seed)
        for n in range(self.number_of_folds):
            data = np.empty((0, 8)) # TODO: Should set the number of columns dinamically
            labels = np.array([])
            sampled_data = {'0': 0, '1': 0}
            for l in number_of_elements_to_sample_by_lable:
                sampled_indexes = np.random.choice(
                    len(data_grouped_by_label[l]), 
                    min(int(number_of_elements_to_sample_by_lable[l]), len(data_grouped_by_label[l])), 
                    replace=False
                )

                sampled_data[l] = min(int(number_of_elements_to_sample_by_lable[l]), len(data_grouped_by_label[l]))

                for i in sampled_indexes:
                    data = np.append(data, [data_grouped_by_label[l][i]], axis=0)
                    labels = np.append(labels, l)
                data_grouped_by_label[l] = np.delete(data_grouped_by_label[l], sampled_indexes, axis=0)
            folds_data.append(data)
            folds_labels.append(labels)

            print(f'Fold {n} created')
            print('Negative percentage: ', sampled_data['0'] / (sampled_data['0'] + sampled_data['1']))
            print('Positive percentage: ', sampled_data['1'] / (sampled_data['0'] + sampled_data['1']))

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