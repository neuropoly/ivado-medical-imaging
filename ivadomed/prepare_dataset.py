import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import PIL
import skimage
import os
import sys
import ivadomed.utils as imed_utils
import scipy


# normalize Image
def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi))


def gaussian_kernel(kernlen=10):
    """
    Returns a 2D Gaussian kernel.

    Args:
        kernlen(int): size of kernel

    Returns:
        array: a 2D array of size (kernlen,kernlen)

    """

    x = np.linspace(-1, 1, kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return normalize(kern2d / kern2d.sum())


def heatmap_generation(image, kernel_size):
    """
    Generate heatmap from image containing sing voxel label using
    convolution with gaussian kernel
    Args:
        image: 2D array containing single voxel label
        kernel_size: size of gaussian kernel

    Returns:
        array: 2D array heatmap matching the label.

    """
    kernel = gaussian_kernel(kernel_size)
    map = scipy.signal.convolve(image, kernel, mode='same')
    return normalize(map)


def add_zero_padding(img_list, x_val=512, y_val=512):
    """
    Add zero padding to each image in an array so they all have matching dimension.
    Args:
        img_list(list): list of input image to pad if a single element is inputed it will change it to a list of len 1
        x_val(int): shape of output alongside x axis
        y_val(int): shape of output alongside y axis

    Returns:
        list: list of padded images the same length as input list
    """
    if type(img_list) != list:
        img_list = [img_list]
    img_zero_padding_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        img_tmp = np.zeros((x_val, y_val), dtype=np.float64)
        img_tmp[0:img.shape[0], 0:img.shape[1]] = img[:, :]
        img_zero_padding_list.append(img_tmp)

    return img_zero_padding_list


def mask2label(path_label, aim='full'):
    """
    Convert nifti image to an array of coordinates
    :param path_label:
    :return:
    Args:
        path_label: path of nifti image
        aim: 'full' or 'c2' full will return all points with label between 3 and 30 , c2 will return only the coordinates of points label 3

    Returns:
        array: array containing the asked point in the format [x,y,z,value]

    """
    image = nib.load(path_label)
    image = nib.as_closest_canonical(image)
    arr = np.array(image.dataobj)
    list_label_image = []
    for i in range(len(arr.nonzero()[0])):
        x = arr.nonzero()[0][i]
        y = arr.nonzero()[1][i]
        z = arr.nonzero()[2][i]
        if aim == 'full':
            if arr[x, y, z] < 30 and arr[x, y, z] != 1:
                list_label_image.append([x, y, z, arr[x, y, z]])
        elif aim == 'c2':
            if arr[x, y, z] == 3:
                list_label_image.append([x, y, z, arr[x, y, z]])
    list_label_image.sort(key=lambda x: x[3])
    return list_label_image


def get_midslice_average(path_im, ind):
    """
    Retrieve the input image for the network. This image is generated by
    averaging the 7 slices in the middle of the volume
    Args:
        path_im(string): path to image
        ind(int): index of the slice around which we will average

    Returns:
        array: an array containing the average image oriented according to RAS+ convention.

    """
    image = nib.load(path_im)
    image = nib.as_closest_canonical(image)
    arr = np.array(image.dataobj)
    numb_of_slice = 3
    # Avoid out of bound error by changing the number of slice taken if needed
    if ind + 3 > arr.shape[0]:
        numb_of_slice = arr.shape[0] - ind
    if ind - numb_of_slice < 0:
        numb_of_slice = ind

    return np.mean(arr[ind - numb_of_slice:ind + numb_of_slice, :, :], 0)


def extract_mid_slice_and_convert_coordinates_to_heatmaps(bids_path, suffix, aim, ap_pad=128, is_pad=320):
    """
     This function takes as input a path to a dataset  and generates two sets of images:
   (i) mid-sagittal image of common size (1,ap_pad,is_pad) and
   (ii) heatmap of disc labels associated with the mid-sagittal image.

    Args:
        bids_path (string): path to BIDS dataset form which images will be generated
        suffix (string): suffix of image that will be processed (e.g., T2w)
        aim(string): 'full' or 'c2'. If 'c2' retrieves only c2 label (value = 3) else create heatmap with all label.
        ap_pad(int): desired output size of second dimension axis which will be
                    achieved from padding small images and cropping bigger ones
        is_pad(int): Desired output size of  3rd dimension axis.


    Returns:
        None. Images are saved in BIDS folder
    """
    t = os.listdir(bids_path)
    t.remove('derivatives')

    for i in range(len(t)):
        sub = t[i]
        path_image = bids_path + t[i] + '/anat/' + t[i] + suffix + '.nii.gz'
        if os.path.isfile(path_image):
            path_label = bids_path + 'derivatives/labels/' + t[i] + '/anat/' + t[i] + suffix \
                         + '_label-disc-manual.nii.gz'
            list_points = mask2label(path_label, aim=aim)
            image_ref = nib.load(path_image)
            nib_ref_can = nib.as_closest_canonical(image_ref)
            imsh = np.array(nib_ref_can.dataobj).shape
            mid = get_midslice_average(path_image, list_points[0][0])
            arr_pred_ref_space = imed_utils.reorient_image(np.expand_dims(mid[:, :], axis=0), 2, image_ref,
                                                           nib_ref_can).astype('float32')
            nib_pred = nib.Nifti1Image(arr_pred_ref_space, image_ref.affine)
            nib.save(nib_pred, bids_path + t[i] + '/anat/' + t[i] + suffix + '_mid.nii.gz')
            lab = nib.load(path_label)
            nib_ref_can = nib.as_closest_canonical(lab)
            label_array = np.zeros(imsh[1:])

            for j in range(len(list_points)):
                label_array[list_points[j][1], list_points[j][2]] = 1

            heatmap = heatmap_generation(label_array[:, :], 10)
            arr_pred_ref_space = imed_utils.reorient_image(np.expand_dims(heatmap[:, :], axis=0), 2, lab, nib_ref_can)
            nib_pred = nib.Nifti1Image(arr_pred_ref_space, lab.affine)
            nib.save(nib_pred,
                     bids_path + 'derivatives/labels/' + t[i] + '/anat/' + t[i] + suffix + '_mid_heatmap.nii.gz')
        else:
            pass
