# -*- coding: utf-8 -*-
"""
Compare metrics from various trained models.

@author: Xinzhe Luo
"""

import os
import pandas as pd
import utils

metrics_path = ['./add_fixed_prediction', './add_moving_prediction', 'add_sym_prediction',
                './comp_fixed_prediction', './comp_moving_prediction', './comp_sym_prediction',
                './diff_fixed_prediction', './diff_moving_prediction', './diff_sym_prediction']

labels = [r'+ & $\nabla F$', r'+ & $\nabla M\circ s$', r'+ & sym',
          r'$\circ$ & $\nabla F$', r'$\circ$ & $\nabla M\circ s$', r'$\circ$ & sym',
          r'diff & $\nabla F$', r'diff & $\nabla M\circ s$', r'diff & sym']

filename = 'metrics.csv'

if __name__ == '__main__':
    # set working directory
    print("Working directory: %s" % os.getcwd())
    os.chdir('../')
    print("Working directory changed to %s" % os.getcwd())

    metrics2compare = []
    for path in metrics_path:
        metrics2compare.append(pd.read_csv(os.path.join(path, filename), index_col=0).to_dict())

    utils.visualise_metrics(metrics2compare, './', save_name='metrics_comparison.png', labels=labels,
                            linewidth=1, markersize=6)
        
