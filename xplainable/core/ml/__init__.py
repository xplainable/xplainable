"""
xplainable.core.ml
"""

import os
import tempfile

# Handle numba compilation issues in Jupyter notebooks
try:
    # Test if we can get the current working directory
    _cwd = os.getcwd()
except (OSError, FileNotFoundError):
    # If we can't get the current working directory, configure numba for safety
    os.environ['NUMBA_CACHE_DIR'] = tempfile.gettempdir()
    os.environ['NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING'] = '1'
    # Try to set a safe working directory
    try:
        os.chdir(tempfile.gettempdir())
    except:
        pass

from .regression import XRegressor
from .classification import XClassifier
