import os

import pytest

HERE = os.path.dirname(__file__)


@pytest.fixture
def tmpdir():
    fpth = os.path.join(f"{HERE}/out_test")
    os.makedirs(fpth, exist_ok=True)
    return fpth


@pytest.fixture()
def test_ini():
    return os.path.join(HERE, "test_files/inference-GW150914_095045.ini")
