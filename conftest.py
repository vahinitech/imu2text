"""Pytest bootstrap: make the repo root importable.

The test modules import the top-level scripts (onhw_models, onhw_seq2seq).
When CI invokes bare `pytest`, the repo root is not on sys.path (unlike
`python -m pytest`, which adds the CWD), so imports fail at collection.
A root conftest.py fixes that for every invocation style.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
