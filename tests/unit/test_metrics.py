from pathlib import Path
import importlib
import numpy
import pytest
import sys
import os


module_dir = os.path.abspath("C:\Programming\Lumed\orpl\src\orpl")
sys.path.append(module_dir)

"""
metrics.py testing module 

"""


# 1. raman_snr

from orpl.metrics import raman_snr

@pytest.mark.metrics
def test_raman_snr_catch_absence_of_input():
    with pytest.raises(TypeError) as e: # Incorrect call returns type error
        raman_snr(), (f"Error was not caugut by the system: {e}")

@pytest.mark.metrics
def test_raman_snr_catch_invalid_input():
    with pytest.raises(AttributeError) as e:
        raman_snr("this", "test", "should", "fail"), (f"Error was not caugut by the system: {e}")

@pytest.mark.metrics
def test_numpy_imports_in_metrics():
    module = importlib.import_module("metrics")
    assert hasattr(module, "np")
    assert module.np is numpy
