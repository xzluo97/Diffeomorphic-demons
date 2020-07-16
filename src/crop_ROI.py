"""
Crop ROI of the data.

"""

import nibabel as nib
import numpy as np
import os


def get_roi_coordinates(image, mag_rate=0.1):
    """
    Produce the cuboid ROI coordinates representing the opposite vertices.

    :param image: the image array
    :param mag_rate: The magnification rate for ROI cropping.
    :return: An array representing the smallest coordinates of ROI;
        an array representing the largest coordinates of ROI.
    """

    arg_index = np.argwhere(image > 0)

    low = np.min(arg_index, axis=0)
    high = np.max(arg_index, axis=0)

    soft_low = np.maximum(np.floor(low - (high - low) * mag_rate / 2), np.zeros_like(low))
    soft_high = np.minimum(np.floor(high + (high - low) * mag_rate / 2), np.asarray(image.shape) - 1)

    return soft_low.astype(np.int32), soft_high.astype(np.int32)


def crop_roi(image_path, mag_rate=0.1, image_suffix='image.nii.gz', label_suffix='label.nii.gz'):
    path, name = os.path.split(image_path)

    nii_image = nib.load(image_path)
    image = nii_image.get_fdata()

    low, high = get_roi_coordinates(image, mag_rate)
    # print(low, high)
    img = nib.Nifti1Image(image[low[0]:high[0], low[1]:high[1], low[2]:high[2]].astype(np.uint16),
                          affine=nii_image.affine, header=nii_image.header)

    nib.save(img, os.path.join(path, 'crop_' + name))

    nii_label = nib.load(image_path.replace(image_suffix, label_suffix))
    label = nii_label.get_fdata()
    lab = nib.Nifti1Image(label[low[0]:high[0], low[1]:high[1], low[2]:high[2]].astype(np.uint8),
                          affine=nii_label.affine, header=nii_label.header)

    nib.save(lab, os.path.join(path, 'crop_' + name.replace(image_suffix, label_suffix)))




if __name__ == '__main__':

    image_paths = ['../data/ADNI_I118671_image.nii.gz', '../data/ADNI_I118673_image.nii.gz']

    for image_path in image_paths:
        crop_roi(image_path)


