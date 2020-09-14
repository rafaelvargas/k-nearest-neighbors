import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import os

if __name__ == '__main__':
    for d in os.listdir('results/csv/'):
        name_without_extension = d.split('.')[0]
        num_folds, num_neighbors = name_without_extension.split('_')
        data = pd.read_csv(f'results/csv/{d}')
        data.plot(y=['Accuracy', 'F1-score'], x='k', ylim=(0, 1), title=f'{num_folds} folds and {num_neighbors} neighbors')
        plt.legend(loc='lower right')
        xticks = ' ,'.join(map(str, range(1, int(num_folds) + 1)))
        tikzplotlib.save(
            f'results/plots/{name_without_extension}.tex', 
            axis_width='10.66cm', 
            axis_height='6cm', 
            extra_axis_parameters=['xtick={' + xticks + '}']
        )