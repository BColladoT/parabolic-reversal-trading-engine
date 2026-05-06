# Manual / integration tests

These are standalone scripts that hit external services (Alpaca API, Ray
cluster) or take more than a few seconds to run. They are excluded from the
default `pytest` collection and therefore from CI.

Run individually with the project venv activated:

```bash
python tests/manual/test_engine.py
python tests/manual/test_connection.py
python tests/manual/test_quick_train.py
python tests/manual/test_env_accounting.py
python tests/manual/test_wfo_data_leakage_complete.py
```

The fast pure-Python tests live one directory up in `tests/` and are run by
CI via `pytest`.
