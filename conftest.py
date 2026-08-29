"""Pytest bootstrap: make the repo root importable.

The tests import the ``imu2text`` package from the source tree rather than an
installed copy. When CI invokes bare `pytest`, the repo root is not on
sys.path (unlike `python -m pytest`, which adds the CWD), so the import fails
at collection. A root conftest.py fixes that for every invocation style.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
