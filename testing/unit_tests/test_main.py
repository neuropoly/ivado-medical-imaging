import pytest

from ivadomed.main import check_multiple_raters

@pytest.mark.parametrize(
    'is_train, loader_params', [
       (False, {"target_suffix":
           [["_seg-axon-manual1", "_seg-axon-manual2"],
            ["_seg-myelin-manual1", "_seg-myelin-manual2"]]
           }) 
])
def test_check_multiple_raters(is_train, loader_params):
    with pytest.raises(SystemExit):
        check_multiple_raters(is_train, loader_params)
