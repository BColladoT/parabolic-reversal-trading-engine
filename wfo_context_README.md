# wfo_context/ has been removed

This directory previously held a snapshot copy of `src/rl/` for isolated WFO experiments. It diverged from `src/rl/` (4 of 5 files had different SHA256s as of 2026-05-17) and nothing in the codebase imported from it.

To recover the snapshot:
```
git checkout wfo-snapshot-2026-05-17 -- wfo_context/
```

For new WFO experiments, use `src/rl/` directly and pass `--config` overrides instead of forking the codebase.
