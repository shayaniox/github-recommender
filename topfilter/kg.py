"""Compatibility shim -> the canonical SED-KGraph module at the repository root.

Loaded by explicit file path rather than `from kg import ...`: this file is itself named
`kg`, so a plain import would resolve back to this module whenever its directory is on
sys.path, producing a circular import.
"""

import importlib.util as _importlib_util
import os as _os

_ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "kg.py")
_spec = _importlib_util.spec_from_file_location("_kg_canonical", _ROOT)
_canonical = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_canonical)

DEFAULT_GRAPH = _canonical.DEFAULT_GRAPH
build_adjacency = _canonical.build_adjacency
expand = _canonical.expand
csv_bidirectional_map = _canonical.csv_bidirectional_map
