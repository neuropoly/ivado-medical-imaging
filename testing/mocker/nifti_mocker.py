import nibabel as nib
import numpy as np
from loguru import logger
from pathlib import Path


def create_mock_nifti1_object(shape=(4, 4, 3), data_type=np.float32):
    """
    Create a mock nifti image.
    """
    x, y, z = shape
    data = np.random.rand(*shape).astype(data_type)
    # data = np.arange(x*y*z).reshape(x,y,z)

    # By convention, nibabel world axes are always in RAS+ orientation (left to Right, posterior to Anterior, inferior to Superior).
    new_image = nib.Nifti1Image(data, affine=np.eye(4))

    return new_image


def create_mock_nifti2_object(shape=(4, 4, 3), data_type=np.float32):
    """
    Create a mock nifti image.
    """
    x, y, z = shape
    data = np.random.rand(*shape).astype(data_type)
    # data = np.arange(x*y*z).reshape(x,y,z)

    # By convention, nibabel world axes are always in RAS+ orientation (left to Right, posterior to Anterior, inferior to Superior).
    new_image = nib.Nifti2Image(data, affine=np.eye(4))

    return new_image


def check_nifty_data(data) -> bool:
    """
    Check if data is a nifty image.
    """
    if isinstance(data, nib.Nifti1Image) or isinstance(data, nib.Nifti2Image):
        logger.info(f"Data Shape: {data.shape}")
        logger.info(f"Data Type: {data.get_data_dtype()}")
        logger.info(f"Affine Shape: {data.affine.shape}")
        logger.info(f"Image Header: {data.header}")
        return True
    else:
        return False


def save_nifty_data(data: nib.Nifti1Image, filename: str or Path):
    """
    Save data to a nifty image.
    """
    nib.save(data, filename)
    return
