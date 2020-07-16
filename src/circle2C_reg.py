# -*- coding: utf-8 -*-
"""
Circle to C registration.

@author: Xinzhe Luo
"""

import os
import numpy as np
import tensorflow as tf
from trainer_3d import _load_data, _one_hot_label, Trainer

target_path = './data/C.png'
source_path = './data/circle.png'

if __name__ == '__main__':
    # set working directory
    print("Working directory: %s" % os.getcwd())
    os.chdir('../')
    print("Working directory changed to %s" % os.getcwd())

    # set saving directory
    model_path = 'circle2C_model_trained'
    prediction_path = 'circle2C_prediction'

    # load data
    target_image = _load_data(target_path, data_format='png', image_size=(592, 592))
    source_image = _load_data(source_path, data_format='png', image_size=(592, 592))

    target_label = _one_hot_label(target_image, labels=(0, 255))
    source_label = _one_hot_label(source_image, labels=(0, 255))

    training_data = {'target_image': target_image, 'target_label': target_label,
                     'source_image': source_image, 'source_label': source_label,
                     'target_affine': np.eye(4), 'target_header': None
                     }


    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # trainer
        trainer = Trainer(input_size=(592, 592, 5), num_channels=1, num_classes=2,
                          demons_type='additive', demons_force='moving', normalization='z-score',
                          max_length=2., exp_steps=7)

        metrics, save_path = trainer.train(training_data, training_iters=500,
                                           display_steps=5,
                                           model_path=model_path,
                                           prediction_path=prediction_path)
