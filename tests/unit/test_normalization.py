import numpy as np
import pytest
import sys
import os



# directory path for --cov

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from orpl.normalization import minmax, maxband, snv, auc


# 1. minmax

@pytest.mark.normalization
def test_minmax_fails_with_no_input():
    with pytest.raises(TypeError):
        minmax()


# 2. maxband

@pytest.mark.normalization
def test_maxband_fails_with_no_input():
    with pytest.raises(TypeError):
        maxband()


# 3. snv

@pytest.mark.normalization
def test_snv_fails_with_no_input():
    with pytest.raises(TypeError):
        snv()


# 4. auc

@pytest.mark.normalization
def test_auc_fails_with_no_input():
    with pytest.raises(TypeError):
        auc()

