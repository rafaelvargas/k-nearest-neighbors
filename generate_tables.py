import pandas as pd
import os 

if __name__ == '__main__':
    for d in os.listdir('results/csv/'):
        name_without_extension = d.split('.')[0]
        num_folds, num_neighbors = name_without_extension.split('_')
        data = pd.read_csv(f'results/csv/{d}')
        means = ['Mean', round(data['Accuracy'].mean(), 3), round(data['F1-score'].mean(), 3)]
        std_devs = ['Std. Dev.', round(data['Accuracy'].std(), 3), round(data['F1-score'].std(), 3)]
        data = data.append(pd.DataFrame([means, std_devs], columns=['k', 'Accuracy', 'F1-score']))
        data.to_latex(
            f'results/tables/{name_without_extension}.tex', 
            index=False, 
            column_format='rcc', 
            caption = f'{num_folds} folds and {num_neighbors} neighbors'
        )