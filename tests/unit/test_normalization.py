from pathlib import Path
import numpy as np
import pytest
from orpl import normalization



@pytest.mark.normalization
def test_minmax_returns_TypeError_when_called_without_arguments():
    with pytest.raises(TypeError) as e: # Incorrect call returns type error
        normalization.minmax(), (f"Error was not caugut by the system: {e}")
