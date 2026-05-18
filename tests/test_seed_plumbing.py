"""Verify --seed flag is wired into train_wfo_quick_test.py."""
import os
os.environ.setdefault("ALPACA_API_KEY", "ci-stub-key")
os.environ.setdefault("ALPACA_SECRET", "ci-stub-secret")
os.environ.setdefault("ALPACA_SECRET_KEY", "ci-stub-secret")

import re
from pathlib import Path


def _read_src() -> str:
    return Path("src/scripts/train_wfo_quick_test.py").read_text(encoding="utf-8")


def test_quick_test_has_seed_flag_in_argparse():
    src = _read_src()
    assert "'--seed'" in src or '"--seed"' in src, "missing --seed argparse declaration"


def test_quick_test_references_args_seed():
    src = _read_src()
    assert "args.seed" in src, "args.seed never read after argparse"


def test_quick_test_propagates_seed_to_torch():
    src = _read_src()
    assert re.search(r"torch\.manual_seed\s*\(", src), "torch.manual_seed not called"


def test_quick_test_propagates_seed_to_numpy():
    src = _read_src()
    assert re.search(r"np\.random\.seed\s*\(|np\.random\.default_rng\s*\(", src), \
        "numpy seeding not present"


def test_quick_test_propagates_seed_to_python_random():
    src = _read_src()
    assert re.search(r"random\.seed\s*\(", src), "stdlib random.seed not called"
