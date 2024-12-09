import nibabel as nib
import torch
import numpy as np
import shutil
import logging
from ivadomed import utils as imed_utils
from ivadomed import inference as imed_inference
from ivadomed import models as imed_models
from pathlib import Path
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


IMAGE_PATH = Path(__data_testing_dir__, "sub-unf01", "anat", "sub-unf01_T1w.nii.gz")
PATH_MODEL = Path(__data_testing_dir__, 'model')
PATH_MODEL_ONNX = Path(PATH_MODEL, 'model.onnx')
PATH_MODEL_PT = PATH_MODEL_ONNX.with_suffix('.pt')
LENGTH_3D = (112, 112, 112)


def test_onnx(download_data_testing_test_files):
    model = imed_models.Modified3DUNet(1, 1)
    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir()
    torch.save(model, PATH_MODEL_PT)
    img = nib.load(IMAGE_PATH).get_fdata().astype('float32')[:16, :64, :32]
    # Add batch and channel dimensions
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    imed_utils.save_onnx_model(model, dummy_input, str(PATH_MODEL_ONNX))

    model = torch.load(PATH_MODEL_PT)
    model.eval()
    out_pt = model(img_tensor).detach().numpy()

    out_onnx = imed_inference.onnx_inference(str(PATH_MODEL_ONNX), img_tensor).numpy()
    shutil.rmtree(PATH_MODEL)
    assert np.allclose(out_pt, out_onnx, rtol=1e-3)


def teardown_function():
    remove_tmp_dir()
