# Transformation to perform on image before network processing
import nibabel as nib
import numpy as np
import ivadomed.loader.tools.utils as imed_loader_utils


def get_midslice_average(path_im, ind, slice_axis=0):
    """
    Extract an average 2D slice out of a 3D volume. This image is generated by
    averaging the 7 slices in the middle of the volume
    Args:
        path_im (string): path to image
        ind (int): index of the slice around which we will average
        slice_axis (int): Slice axis according to RAS convention

    Returns:
        nifti: a single slice nifti object containing the average image in the image space.

    """
    image = nib.load(path_im)
    image_can = nib.as_closest_canonical(image)
    arr_can = np.array(image_can.dataobj)
    numb_of_slice = 3
    # Avoid out of bound error by changing the number of slice taken if needed
    if ind + 3 > arr_can.shape[slice_axis]:
        numb_of_slice = arr_can.shape[slice_axis] - ind
    if ind - numb_of_slice < 0:
        numb_of_slice = ind

    slc = [slice(None)] * len(arr_can.shape)
    slc[slice_axis] = slice(ind - numb_of_slice, ind + numb_of_slice)
    mid = np.mean(arr_can[tuple(slc)], slice_axis)

    arr_pred_ref_space = imed_loader_utils.reorient_image(np.expand_dims(mid[:, :], axis=slice_axis), 2, image,
                                                   image_can).astype('float32')
    nib_pred = nib.Nifti1Image(arr_pred_ref_space, image.affine)

    return nib_pred
