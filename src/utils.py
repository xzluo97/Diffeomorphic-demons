# -*- coding: utf-8 -*-
"""
Functions and operations for image re-sampling and result visualization.

@author: Xinzhe Luo
"""

import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import logging
import itertools
import math
import cv2
import matplotlib.pyplot as plt


def normalize_image(image, normalization=None):
    if normalization == 'min-max':
        image -= tf.reduce_min(image)
        image /= tf.reduce_max(image)

    elif normalization == 'z-score':
        image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)

    return image


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*3)
        k = tf.exp([-0.5*x**2/sigma**2 for x in range(-tail, tail+1)])
        return k / tf.reduce_sum(k)


def separable_gaussian_filter3d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        channels = vol.get_shape().as_list()[-1]
        strides = [1, 1, 1, 1, 1]

        return tf.concat([tf.nn.conv3d(tf.nn.conv3d(tf.nn.conv3d(
            vol[..., i, None],
            tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, -1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, 1, -1, 1, 1]), strides, "SAME") for i in range(channels)], axis=-1)


def separable_gaussian_filter2d(vol, kernel):
    if kernel == 0:
        return vol
    else:
        channels = vol.get_shape().as_list()[-1]
        strides = [1, 1, 1, 1]

        return tf.concat([tf.nn.conv2d(tf.nn.conv2d(
            vol[..., i, None],
            tf.reshape(kernel, [-1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, -1, 1, 1]), strides, "SAME") for i in range(channels)], axis=-1)


def compute_image_gradient(image, dim=3):
    """
    Compute the image gradient by finite difference.

    :param image: The image intensity tensor, of shape [batch, nx, ny, nz, channels].
    :return: A tensor of image gradient of the same shape as the image tensor. The gradient values are organized so that
             [I(x+1, y, z)-I(x-1, y, z)] / 2 is in location (x, y, z), and the result is padded to match the size of the
             input image. Output shape: [batch, nx, ny, nz, channels * 3].
    """
    with tf.name_scope('image_gradient'):
        channels = image.get_shape().as_list()[-1]
        def gradient_ixyz(image, fn):
            return tf.stack([fn(image[..., i]) for i in range(channels)], axis=-1)

        if dim == 3:
            def gradient_dx(fv):
                return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

            def gradient_dy(fv):
                return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

            def gradient_dz(fv):
                return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

            dIdx = gradient_ixyz(image, gradient_dx)  # [batch, nx-2, ny-2, nz-2, channels]
            dIdy = gradient_ixyz(image, gradient_dy)
            dIdz = gradient_ixyz(image, gradient_dz)

            gradient = tf.pad(tf.concat([dIdx, dIdy, dIdz], axis=-1), paddings=[[0, 0],
                                                                                [1, 1], [1, 1], [1, 1],
                                                                                [0, 0]],
                              name='image_gradient')

        elif dim == 2:
            def gradient_dx(fv):
                return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2

            def gradient_dy(fv):
                return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

            dIdx = gradient_ixyz(image, gradient_dx)  # [batch, nx-2, ny-2, channels]
            dIdy = gradient_ixyz(image, gradient_dy)
            gradient = tf.pad(tf.concat([dIdx, dIdy], axis=-1), paddings=[[0, 0],
                                                                          [1, 1], [1, 1],
                                                                          [0, 0]],
                              name='image_gradient')

        return gradient


def get_segmentation(predictor, mode='tf'):
    """
    produce the segmentation maps from the probability maps
    """
    assert mode in ['tf', 'np'], "The mode must be either 'tf' or 'np'!"
    if mode == 'tf':
        assert isinstance(predictor, tf.Tensor)
        return tf.where(tf.equal(tf.reduce_max(predictor, -1, keepdims=True), predictor),
                        tf.ones_like(predictor),
                        tf.zeros_like(predictor))

    elif mode == 'np':
        assert isinstance(predictor, np.ndarray)
        return np.where(np.equal(np.max(predictor, -1, keepdims=True), predictor),
                        np.ones_like(predictor),
                        np.zeros_like(predictor))


def save_prediction_png(image, label, pred, save_path, **kwargs):
    """
    Combine each prediction and the corresponding ground truth as well as input into one image and save as png files.

    :param image: The raw image array.
    :param label: The one-hot ground-truth array.
    :param pred: The prediction array.
    :param save_path: where to save the validation/test predictions, has the form of 'directory'
    :param kwargs: save_name - The saved filename if given;
                   save_prefix - The prefix of the saved filename;
                   slice_indices - List of slice indices for visualization.
    """
    save_name = kwargs.pop("save_name", 'prediction.png')
    save_prefix = kwargs.pop("save_prefix", '')
    slice_indices = kwargs.pop("slice_indices", None)

    abs_pred_path = os.path.abspath(save_path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    pred = np.where(np.equal(np.max(pred, -1, keepdims=True), pred),
                    np.ones_like(pred), np.zeros_like(pred))

    plt.imsave(os.path.join(save_path, save_prefix + save_name),
               combine_img_prediction(image, label, pred, slice_indices))


def save_prediction_nii(pred, save_path, data_type='image', **kwargs):
    """
    Save the predictions into nifty images.
    Predictions are pre-processed by setting the maximum along classes to be a certain intensity specific to its class.

    :param pred: The prediction array of shape [*vol_shape, n_class].
    :param save_path: where to save the validation/test predictions, has the form of 'directory'
    :param data_type: 'image' or 'label'
    :param kwargs: save_name - The saved filename if given;
                   affine: The affine matrix array that relates array coordinates from the image data array to 
                        coordinates in some RAS+ world coordinate system.
                   header - The header that contains the image metadata;
                   save_prefix - The prefix of the saved filename;
                   save_suffix - The suffix of the saved filename
                   original_size - The original size of the input for padding

    """
    
    intensities = kwargs.pop("intensities", range(pred.shape[-1]))
    affine = kwargs.pop("affine", np.eye(4))
    header = kwargs.pop("header", None)
    save_prefix = kwargs.pop("save_prefix", '')
    save_suffix = kwargs.pop("save_suffix", '.nii.gz')
    original_size = kwargs.pop("original_size", None)

    if original_size is None:
        original_size = pred.shape

    abs_pred_path = os.path.abspath(save_path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)
    
    if data_type == 'image':
        if pred.ndim == 4:
            pred = np.mean(pred, axis=-1)
        save_name = kwargs.pop("save_name", 'warped_image')
        img = nib.Nifti1Image(pred.astype(np.uint16), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + save_name + save_suffix))

    elif data_type == 'jacobian':
        save_name = kwargs.pop("save_name", 'jacobian_det')
        pred -= pred.min()
        pred /= pred.max()
        pred *= 1000
        img = nib.Nifti1Image(pred.astype(np.uint16), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + save_name + save_suffix))

    elif data_type == 'vector_fields':
        save_name = kwargs.pop("save_name", 'vector_fields')
        if pred.shape[-1] <= 2:
            zero_fields = np.zeros([pred.shape[0], pred.shape[1], pred.shape[2], 3 - pred.shape[-1]])
            pred = np.concatenate([pred, zero_fields], axis=-1)
        img = nib.Nifti1Image(pred.astype(np.float32), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + save_name + save_suffix))
        
    elif data_type == 'label':
        save_name = kwargs.pop('save_name', 'warped_label')
        class_preds = []
        for i in range(pred.shape[-1]):
            if i == 0:
                class_preds.append(np.pad(pred[..., i],
                                          (((original_size[0] - pred.shape[0]) // 2,
                                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                                           ((original_size[1] - pred.shape[1]) // 2,
                                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                                           ((original_size[2] - pred.shape[2]) // 2,
                                            original_size[2] - pred.shape[2] - (original_size[2] - pred.shape[2]) // 2)
                                           ), 'constant', constant_values=1))
            else:
                class_preds.append(np.pad(pred[..., i],
                                          (((original_size[0] - pred.shape[0]) // 2,
                                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                                           ((original_size[1] - pred.shape[1]) // 2,
                                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                                           ((original_size[2] - pred.shape[2]) // 2,
                                            original_size[2] - pred.shape[2] - (original_size[2] - pred.shape[2]) // 2)
                                           ), 'constant'))
    
        pred = np.stack(class_preds, -1)
    
        intensities = np.tile(np.asarray(intensities), np.concatenate((pred.shape[:-1], [1])))
        mask = np.equal(np.max(pred, -1, keepdims=True), pred)
        img = nib.Nifti1Image(np.sum(mask * intensities, axis=-1).astype(np.uint16), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + save_name + save_suffix))


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [n, *vol_shape, channels]

    :returns img: the rgb image [n, *vol_shape, 3]
    """
    if len(img.shape) < 5:
        img = np.expand_dims(img, axis=-1)
    channels = img.shape[-1]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    for k in range(np.shape(img)[3]):
        st = img[:, :, :, k, ]
        if np.amin(st) != np.amax(st):
            st -= np.amin(st)
            st /= np.amax(st)
        st *= 255
    return img.round().astype(np.uint8)


def combine_img_prediction(data, gt, pred, slice_indices=None, slice_axis='z'):
    """
    Combines the data, ground truth and the prediction into one rgb image for each class

    :param data: the data tensor, of shape [1, *vol_shape, 1]
    :param gt: the ground truth tensor, of shape [1, *vol_shape, n_class]
    :param pred: the prediction tensor, of shape [1, *vol_shape, n_class]
    :param slice_indices: List of slice indices for visualization.
    :param slice_axis: The slicing axis.

    :returns: the concatenated rgb image
    """
    axis_index = {'x': 1, 'y': 2, 'z': 3}

    if slice_indices is None:
        slice_indices = list(range(0, data.shape[axis_index[slice_axis]], 2))

    assert all([i < pred.shape[axis_index[slice_axis]] for i in slice_indices]), "The slicing index exceeds " \
                                                                                 "the dimension!"

    image_slice = np.take(data, indices=slice_indices, axis=axis_index[slice_axis])
    label_slice = np.take(gt, indices=slice_indices, axis=axis_index[slice_axis])
    pred_slice = np.take(pred, indices=slice_indices, axis=axis_index[slice_axis])

    image_column = np.concatenate([x.squeeze(axis_index[slice_axis])
                                   for x in np.split(to_rgb(crop_to_shape(image_slice, pred_slice.shape)),
                                                     len(slice_indices),
                                                     axis=axis_index[slice_axis])],
                                  axis=1)

    n_class = gt.shape[-1]
    class_labels = []
    class_preds = []
    for k in range(n_class):
        class_labels.append(np.concatenate([x.squeeze(axis_index[slice_axis])
                                            for x in np.split(dye_label(to_rgb(crop_to_shape(label_slice[..., k],
                                                                                             pred_slice.shape)),
                                                                        class_index=k),
                                                              len(slice_indices),
                                                              axis=axis_index[slice_axis])],
                                           axis=1))

        class_preds.append(np.concatenate([x.squeeze(axis_index[slice_axis])
                                           for x in np.split(dye_label(to_rgb(pred_slice[..., k]),
                                                                       class_index=k),
                                                             len(slice_indices),
                                                             axis=axis_index[slice_axis])],
                                          axis=1))

    # print(np.sum(np.stack(class_labels), axis=0).shape)
    # print(image_column.shape)

    label_column = cv2.addWeighted(np.sum(np.stack(class_labels), axis=0).astype(np.uint8), 0.8, image_column, 1, 0)
    pred_column = cv2.addWeighted(np.sum(np.stack(class_preds), axis=0).astype(np.uint8), 0.8, image_column, 1, 0)

    final = np.squeeze(np.concatenate([image_column, label_column, pred_column], axis=2), axis=0)
    return final


def dye_label(label, class_index):
    """
    Dye the label with colors.

    :param label: The RGB one-hot label of shape [1, *vol_shape, 3].
    :param class_index: The class index.
    :return: The colorized label map.
    Todo:
        Enable customized RGB values.
    """
    rgb_values = [[0, 0, 0], [255, 250, 205], [188, 143, 143], [199, 21, 133],
                  [188, 143, 143], [135, 206, 235], [238, 130, 238], [253, 245, 230]]

    label[..., 0][label[..., 0] == 255] = rgb_values[class_index][0]
    label[..., 1][label[..., 1] == 255] = rgb_values[class_index][1]
    label[..., 2][label[..., 2] == 255] = rgb_values[class_index][2]

    return label

def crop_to_shape(data, shape, mode='np'):
    """
    Crops the volumetric tensor or array into the given image shape by removing the border
    (expects a tensor or array of shape [n_batch, *vol_shape, channels]).

    :param data: the tensor or array to crop, shape=[n_batch, *vol_shape, n_class]
    :param shape: the target shape
    :param mode: 'np' or 'tf'.
    :return: The cropped tensor or array.
    """
    assert mode in ['np', 'tf'], "The mode must be either 'np' or 'tf'!"
    if mode == 'np':
        data_shape = data.shape
    elif mode == 'tf':
        data_shape = data.get_shape().as_list()

    if len(shape) <= 3:
        shape = (1, ) + shape + (1, )

    assert np.all(tuple(data_shape[1:4]) >= shape[1:4]), "The shape of array to be cropped is smaller than the " \
                                                         "target shape."
    offset0 = (data_shape[1] - shape[1]) // 2
    offset1 = (data_shape[2] - shape[2]) // 2
    offset2 = (data_shape[3] - shape[3]) // 2
    remainder0 = (data_shape[1] - shape[1]) % 2
    remainder1 = (data_shape[2] - shape[2]) % 2
    remainder2 = (data_shape[3] - shape[3]) % 2

    if (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) == 0:
        return data

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) == 0:
        return data[:, offset0:(-offset0 - remainder0), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) != 0 and (data_shape[3] - shape[3]) == 0:
        return data[:, :, offset1:(-offset1 - remainder1), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) != 0:
        return data[:, :, :, offset2:(-offset2 - remainder2), ]

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) != 0 and (data_shape[3] - shape[3]) == 0:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) != 0 and (data_shape[3] - shape[3]) != 0:
        return data[:, :, offset1:(-offset1 - remainder1), offset2:(-offset2 - remainder2), ]

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) == 0 and (data_shape[3] - shape[3]) != 0:
        return data[:, offset0:(-offset0 - remainder0), :, offset2:(-offset2 - remainder2), ]

    else:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1),
               offset2:(-offset2 - remainder2), ]


def visualise_metrics(metrics, save_path, **kwargs):
    """
    Visualise training and test metrics for comparison.

    :param metrics: A list of metrics from different datasets for visualisation.
    :param save_path: Where to save the figure.
    :return: None.
    """
    if not isinstance(metrics, (list, tuple)):
        metrics = list(metrics)

    assert len(metrics) <= 9, "Number of types of metrics should be less than 8."
    assert checkEqual([m.keys() for m in metrics]), "Types of metrics must be equal among all datasets."

    labels = kwargs.pop('labels', ['dataset %s' % k for k in range(len(metrics))])
    linewidth = kwargs.pop('linewidth', 1)
    if isinstance(linewidth, (int, float)):
        linewidth = [linewidth] * len(metrics)

    markersize = kwargs.pop('markersize', 12)
    if isinstance(markersize, (int, float)):
        markersize = [markersize] * len(metrics)

    metric_types = list(metrics[0].keys())
    n_rows, n_cols = factor_int(len(metric_types))
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(12 * n_cols, 6 * n_rows))
    fig.canvas.set_window_title('Visualisation of metrics from various datasets.')

    plt.plot()
    colors = ['b','k', 'gray', 'g', 'c', 'm', 'orange', 'yellow', 'r']
    markers = ['.', 'o', 'v', '^', '<', '>', '+', 'x', '*']

    for i in range(n_rows):
        for j in range(n_cols):
            type = metric_types[i * n_cols + j]
            max_x = np.max(list(itertools.chain(*[list(m[type].keys()) for m in metrics])))
            max_y = np.max(list(itertools.chain(*[list(m[type].values()) for m in metrics])))
            min_y = np.min(list(itertools.chain(*[list(m[type].values()) for m in metrics])))

            for k in range(len(metrics)):
                m = metrics[k]
                # extract data
                x, y = list(m[type].keys()), list(m[type].values())

                # plot metrics
                ax[i, j].plot(x, y, linestyle='-', color=colors[k], marker=markers[k], label=labels[k],
                              linewidth=linewidth[k], markersize=markersize[k])

            # set title
            ax[i, j].set_title(type)
            # set label
            ax[i, j].set_xlabel('Iterations')
            ax[i, j].set_ylabel('Values')
            # set grid
            ax[i, j].grid(b='true', which='both', axis='both', color='gray')
            # set limits
            ax[i, j].set_xlim(-1, max_x * 1.1)
            ax[i, j].set_ylim(min(min_y * 1.1, 0), max_y * 1.1)

            # set legend
            ax[i, j].legend(loc='upper left')

    save_name = kwargs.pop('save_name', "metrics_visualisation.png")
    plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight', dpi=300)


def factor_int(n):
    """
    Factorize an integer into factors as close to a squared root as possible.

    :param n: The given integer.
    :return: The two factors.
    """
    nsqrt = math.ceil(math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n / val)
        if val2 * val == float(n):
            solution = True
        else:
            val -= 1
    return val, val2


def checkEqual(lst):
    return lst[1:] == lst[:-1]
