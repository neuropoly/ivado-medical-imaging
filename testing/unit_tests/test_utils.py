import string
from ivadomed.utils import get_timestamp, get_win_system_memory, get_linux_system_memory, get_mac_system_memory
from loguru import logger
import platform
import pytest

current_platform = platform.system()


def test_timestamp():
    """
    Test the timestamp function.
    """
    output = get_timestamp()
    logger.debug(output)
    assert output.count("-") == 2
    assert output.count(".") == 1
    assert output.count("T") == 1
    for I in string.ascii_uppercase:
        if I == "T":
            assert output.count(I) == 1
        else:
            assert output.count(I) == 0
    for i in string.ascii_lowercase:
        assert output.count(i) == 0

@pytest.mark.skipif(current_platform != "Windows", reason="Function only works for Windows, skip on all other OS")
def test_get_win_system_memory():
    """
    Get Windows memory size
    Returns:

    """
    # Most computers/clusters should have memory of at least 100mb and no more than 256GB RAM
    assert 0.1 < get_win_system_memory() < 256

@pytest.mark.skipif(current_platform != "Linux", reason="Function only works for Linux, skip on all other OS")
def test_get_linux_system_memory():
    """
    Get Windows memory size
    Returns:

    """
    # Most computers/clusters should have memory of at least 100mb and no more than 256GB RAM
    assert 0.1 < get_linux_system_memory() < 256
@pytest.mark.skipif(current_platform != "Darwin", reason="Function only works for Mac, skip on all other OS")
def test_get_mac_system_memory():
    """
    Get Windows memory size
    Returns:

    """
    # Most computers/clusters should have memory of at least 100mb and no more than 256GB RAM
    assert 0.1 < get_mac_system_memory() < 256