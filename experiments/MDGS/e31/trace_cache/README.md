The E31 scripts look for local `train.jsonl`, `val.jsonl`, and `test.jsonl` files here first.

If they are absent, the loader falls back to the existing sibling E29 trace corpus at `../e29/trace_cache/`.
