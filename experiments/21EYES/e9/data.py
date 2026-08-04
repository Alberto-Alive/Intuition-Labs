from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_E6_DATA = Path(__file__).resolve().parents[1] / "e6" / "data.py"
_SPEC = importlib.util.spec_from_file_location("_e6_synthetic_data", _E6_DATA)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load E6 data generator from {_E6_DATA}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


PAD = _MODULE.PAD
NOISE = _MODULE.NOISE
SET = _MODULE.SET
DISTRACTOR = _MODULE.DISTRACTOR
IGNORE = _MODULE.IGNORE
QUERY = _MODULE.QUERY
QMARK = _MODULE.QMARK
ASSIGN = _MODULE.ASSIGN
EQ = _MODULE.EQ
ASK = _MODULE.ASK
ARROW = _MODULE.ARROW
REALSET = _MODULE.REALSET
FAKESET = _MODULE.FAKESET
OVERWRITE = _MODULE.OVERWRITE
FAKEOVERWRITE = _MODULE.FAKEOVERWRITE
LINK = _MODULE.LINK
FAKELINK = _MODULE.FAKELINK

KEY_OFFSET = _MODULE.KEY_OFFSET
N_KEYS = _MODULE.N_KEYS
VALUE_OFFSET = _MODULE.VALUE_OFFSET
N_VALUES = _MODULE.N_VALUES
VOCAB_SIZE = _MODULE.VOCAB_SIZE

TASKS = _MODULE.TASKS
TASK_TO_ID = _MODULE.TASK_TO_ID

Statement = _MODULE.Statement
SyntheticExample = _MODULE.SyntheticExample
SyntheticBatch = _MODULE.SyntheticBatch
SyntheticBatcher = _MODULE.SyntheticBatcher

key_token = _MODULE.key_token
value_token = _MODULE.value_token
token_name = _MODULE.token_name
is_heldout_pair = _MODULE.is_heldout_pair
generate_example = _MODULE.generate_example
batch_examples = _MODULE.batch_examples
make_batch = _MODULE.make_batch


__all__ = [
    "PAD",
    "NOISE",
    "SET",
    "DISTRACTOR",
    "IGNORE",
    "QUERY",
    "QMARK",
    "ASSIGN",
    "EQ",
    "ASK",
    "ARROW",
    "REALSET",
    "FAKESET",
    "OVERWRITE",
    "FAKEOVERWRITE",
    "LINK",
    "FAKELINK",
    "KEY_OFFSET",
    "N_KEYS",
    "VALUE_OFFSET",
    "N_VALUES",
    "VOCAB_SIZE",
    "TASKS",
    "TASK_TO_ID",
    "Statement",
    "SyntheticExample",
    "SyntheticBatch",
    "SyntheticBatcher",
    "key_token",
    "value_token",
    "token_name",
    "is_heldout_pair",
    "generate_example",
    "batch_examples",
    "make_batch",
]
