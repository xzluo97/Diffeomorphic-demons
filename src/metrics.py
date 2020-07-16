# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import utils


class OverlapMetrics(object):
    """
    Compute the Dice similarity coefficient between the ground truth and the prediction.

    """

    def __init__(self, n_class=1, eps=0.1, mode='tf'):

        self.n_class = n_class
        self.eps = eps
        self.mode = mode

        assert mode in ['tf', 'np'], "The mode must be either 'tf' or 'np'!"

    def averaged_foreground_dice(self, y_true, y_seg):
        """
        Assume the first class is the background.
        """
        if self.mode == 'tf':
            assert y_true.shape[1:] == y_seg.shape[1:], "The ground truth and prediction must be of equal shape! " \
                                                        "Ground truth shape: %s, " \
                                                        "prediction shape: %s" % (y_true.get_shape().as_list(),
                                                                                  y_seg.get_shape().as_list())

            assert y_seg.get_shape().as_list()[-1] == self.n_class, "The number of classes of the segmentation " \
                                                                    "should be equal to %s!" % self.n_class
        elif self.mode == 'np':
            assert y_true.shape == y_seg.shape, "The ground truth and prediction must be of equal shape! " \
                                                "Ground truth shape: %s, prediction shape: %s" % (y_true.shape,
                                                                                                  y_seg.shape)

            assert y_seg.shape[-1], "The number of classes of the segmentation should be equal to %s!" % self.n_class

        y_seg = utils.get_segmentation(y_seg, self.mode)
        dice = 0.
        if self.mode == 'tf':
            for i in range(1, self.n_class):
                top = 2 * tf.reduce_sum(y_true[..., i] * y_seg[..., i])
                bottom = tf.reduce_sum(y_true[..., i] + y_seg[..., i])
                dice += top / (tf.maximum(bottom, self.eps))

            return tf.divide(dice, tf.cast(self.n_class - 1, dtype=tf.float32), name='averaged_foreground_dice')

        elif self.mode == 'np':
            for i in range(1, self.n_class):
                top = 2 * np.sum(y_true[..., i] * y_seg[..., i])
                bottom = np.sum(y_true[..., i] + y_seg[..., i])
                dice += top / (np.maximum(bottom, self.eps))

            return np.divide(dice, self.n_class - 1)

    def class_specific_dice(self, y_true, y_seg, i):
        """
        Compute the class specific Dice.

        :param i: The i-th tissue class, default parameters: 0 for background; 1 for myocardium of the left ventricle;
            2 for left atrium; 3 for left ventricle; 4 for right atrium; 5 for right ventricle; 6 for ascending aorta;
            7 for pulmonary artery.
        """
        y_seg = utils.get_segmentation(y_seg, self.mode)

        if self.mode == 'tf':
            assert y_true.shape[1:] == y_seg.shape[1:], "The ground truth and prediction must be of equal shape! " \
                                                        "Ground truth shape: %s, " \
                                                        "prediction shape: %s" % (y_true.get_shape().as_list(),
                                                                                  y_seg.get_shape().as_list())

            top = 2 * tf.reduce_sum(y_true[..., i] * y_seg[..., i])
            bottom = tf.reduce_sum(y_true[..., i] + y_seg[..., i])
            dice = tf.divide(top, tf.maximum(bottom, self.eps), name='class%s_dice' % i)

        elif self.mode == 'np':
            assert y_true.shape == y_seg.shape, "The ground truth and prediction must be of equal shape! " \
                                                "Ground truth shape: %s, prediction shape: %s" % (y_true.shape,
                                                                                                  y_seg.shape)

            top = 2 * np.sum(y_true[..., i] * y_seg[..., i])
            bottom = np.sum(y_true[..., i] + y_seg[..., i])
            dice = np.divide(top, np.maximum(bottom, self.eps))

        return dice

    def averaged_foreground_jaccard(self, y_true, y_seg):
        """
                Assume the first class is the background.
                """
        if self.mode == 'tf':
            assert y_true.shape[1:] == y_seg.shape[1:], "The ground truth and prediction must be of equal shape! " \
                                                        "Ground truth shape: %s, " \
                                                        "prediction shape: %s" % (y_true.get_shape().as_list(),
                                                                                  y_seg.get_shape().as_list())

            assert y_seg.get_shape().as_list()[-1] == self.n_class, "The number of classes of the segmentation " \
                                                                    "should be equal to %s!" % self.n_class
        elif self.mode == 'np':
            assert y_true.shape == y_seg.shape, "The ground truth and prediction must be of equal shape! " \
                                                "Ground truth shape: %s, prediction shape: %s" % (y_true.shape,
                                                                                                  y_seg.shape)

            assert y_seg.shape[-1], "The number of classes of the segmentation should be equal to %s!" % self.n_class

        y_seg = utils.get_segmentation(y_seg, self.mode)
        jaccard = 0.
        if self.mode == 'tf':
            y_true = tf.cast(y_true, dtype=tf.bool)
            y_seg = tf.cast(y_seg, dtype=tf.bool)
            for i in range(1, self.n_class):
                top = tf.reduce_sum(tf.cast(tf.logical_and(y_true[..., i], y_seg[..., i]), tf.float32))
                bottom = tf.reduce_sum(tf.cast(tf.logical_or(y_true[..., i], y_seg[..., i]), tf.float32))
                jaccard += top / (tf.maximum(bottom, self.eps))

            return tf.divide(jaccard, tf.cast(self.n_class - 1, dtype=tf.float32),
                             name='averaged_foreground_jaccard')

        elif self.mode == 'np':
            y_true = y_true.astype(np.bool)
            y_seg = y_seg.astype(np.bool)
            for i in range(1, self.n_class):
                top = np.sum(np.logical_and(y_true[..., i], y_seg[..., i]).astype(np.float32))
                bottom = np.sum(np.logical_or(y_true[..., i], y_seg[..., i]).astype(np.float32))
                jaccard += top / (np.maximum(bottom, self.eps))

            return np.divide(jaccard, self.n_class - 1)


def local_displacement_energy(ddf, energy_type, energy_weight=1.):
    """
    Compute the displacement energy for regularization.

    :param ddf: The 3-D dense displacement fields, of shape [n_batch, *vol_shape, n_atlas, 3].
    :param energy_type: The energy type, chosen from ['bending', 'gradient-l1', 'gradient-l2'].
    :param energy_weight: The energy weight.
    :return: A scalar tensor representing the computed displacement energy.
    """
    with tf.name_scope('displacement_energy'):
        n_channel = ddf.get_shape().as_list()[-1]

        def gradient_dx(fv):
            return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

        def gradient_dy(fv):
            return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

        def gradient_dz(fv):
            return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

        def gradient_txyz(Txyz, fn):
            return tf.stack([fn(Txyz[..., i]) for i in range(n_channel)], axis=-1)

        def compute_gradient_norm(displacement, flag_l1=False):
            dTdx = gradient_txyz(displacement, gradient_dx)
            dTdy = gradient_txyz(displacement, gradient_dy)
            dTdz = gradient_txyz(displacement, gradient_dz)
            if flag_l1:
                norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
            else:
                norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
            return tf.reduce_mean(norms)

        def compute_bending_energy(displacement):
            dTdx = gradient_txyz(displacement, gradient_dx)
            dTdy = gradient_txyz(displacement, gradient_dy)
            dTdz = gradient_txyz(displacement, gradient_dz)
            dTdxx = gradient_txyz(dTdx, gradient_dx)
            dTdyy = gradient_txyz(dTdy, gradient_dy)
            dTdzz = gradient_txyz(dTdz, gradient_dz)
            dTdxy = gradient_txyz(dTdx, gradient_dy)
            dTdyz = gradient_txyz(dTdy, gradient_dz)
            dTdxz = gradient_txyz(dTdx, gradient_dz)
            return tf.reduce_mean(
                dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)

        if energy_weight:
            if energy_type == 'bending':
                energy = compute_bending_energy(ddf)
            elif energy_type == 'gradient-l2':
                energy = compute_gradient_norm(ddf)
            elif energy_type == 'gradient-l1':
                energy = compute_gradient_norm(ddf, flag_l1=True)
            else:
                raise Exception('Not recognised local regulariser!')
        else:
            energy = tf.constant(0.0)

        return tf.multiply(energy, energy_weight, name='displacement_energy')


def compute_jacobian_determinant(vector_fields):
    """
    Compute the average Jacobian determinants of the vector fields.

    :param vector_fields: The vector fields of shape [batch, nx, ny, nz, 3].
    :return: The Jacobian determinant of the vector fields of shape [batch, nx, ny, nz].
    """
    with tf.name_scope('vector_jacobian_determinant'):
        n_dims = vector_fields.get_shape().as_list()[-1]
        def gradient_dx(fv):
            return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

        def gradient_dy(fv):
            return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

        def gradient_dz(fv):
            return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

        def gradient_txyz(Txyz, fn):
            return tf.stack([fn(Txyz[..., i]) for i in range(n_dims)])

        dTdx = gradient_txyz(vector_fields, gradient_dx)  # [3, batch, nx, ny, nz]
        dTdy = gradient_txyz(vector_fields, gradient_dy)
        dTdz = gradient_txyz(vector_fields, gradient_dz)

        jacobian_det = tf.subtract((dTdx[0]+1)*(dTdy[1]+1)*(dTdz[2]+1) + dTdx[2]*dTdy[0]*dTdz[1] + dTdx[1]*dTdy[2]*dTdz[0],
                                   dTdx[2]*(dTdy[1]+1)*dTdz[0] + (dTdx[0]+1)*dTdy[2]*dTdz[1] + dTdx[1]*dTdy[0]*(dTdz[2]+1),
                                   name='jacobian_det')

        return jacobian_det


if __name__ == '__main__':
    import os
    from trainer_3d import _load_data, _one_hot_label

    print("Working directory: %s" % os.getcwd())
    os.chdir('../')
    print("Working directory changed to %s" % os.getcwd())

    target_label = _one_hot_label(_load_data('./data/crop_ADNI_I118671_label.nii.gz', clip_value=False)[0],
                                  labels=[0., 8., 9., 19., 20., 25., 26., 27., 28.])

    source_label = _one_hot_label(_load_data('./data/affine_crop_ADNI_I118673_label.nii.gz', clip_value=False)[0],
                                  labels=[0., 8., 9., 19., 20., 25., 26., 27., 28.])

    dice = OverlapMetrics(n_class=9, mode='np').averaged_foreground_dice(y_true=target_label, y_seg=source_label)

    print(dice)
