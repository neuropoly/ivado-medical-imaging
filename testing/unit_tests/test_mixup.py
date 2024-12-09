import ivadomed.mixup as imed_mixup
import torch
import pytest
import logging
from testing.unit_tests.t_utils import create_tmp_dir,  __tmp_dir__
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize("debugging", [False, True])
@pytest.mark.parametrize("ofolder", [str(Path(__tmp_dir__, "test")),
                                     str(Path(__tmp_dir__, "mixup_test"))])
def test_mixup(debugging, ofolder):
    inp = [[[[0 for i in range(40)] for i in range(40)]]]
    targ = [[[[0 for i in range(40)] for i in range(40)]]]
    for i in range(10):
        for j in range(10):
            targ[0][0][i][j] = 1
    inp = torch.tensor(inp).float()
    targ = torch.tensor(targ).float()
    # just testing if mixup function run
    imed_mixup.mixup(inp, targ, alpha=0.5, debugging=debugging, ofolder=ofolder)


def teardown_function():
    remove_tmp_dir()
