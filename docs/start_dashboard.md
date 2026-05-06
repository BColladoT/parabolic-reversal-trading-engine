# Starting the Training Dashboard UI

## Command

```bash
python src/dashboard/serve.py --metrics-dir models/wfo_test --port 8050
```

This starts the WFO Training Dashboard & Command Center at **http://localhost:8050**.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--metrics-dir` | `models/wfo_test` | Directory containing `training_metrics.jsonl` and `trades.jsonl` |
| `--port` | `8050` | Server port |
| `--no-open` | off | Don't auto-open browser on start |

## Examples

```bash
# Default (quick test metrics, port 8050, auto-opens browser)
python src/dashboard/serve.py

# Full WFO metrics on a different port
python src/dashboard/serve.py --metrics-dir models/wfo --port 8051

# Without auto-opening browser
python src/dashboard/serve.py --metrics-dir models/wfo_test --no-open
```

## What It Provides

- **Dashboard UI** at `/` — real-time training charts and trade summaries
- **Command Center** — start/stop training runs (custom or full WFO) directly from the UI
- **API endpoints**:
  - `GET /api/metrics` — training metrics (loss, reward, etc.)
  - `GET /api/trades` — individual trade log
  - `GET /api/trades-summary` — aggregated trade statistics by phase
  - `GET /api/test-results` — final WFO fold results
  - `GET /api/config` — training config and fold info
  - `GET /api/status` — training process status
  - `GET /api/process-log` — training process stdout/stderr
  - `GET /api/data-range` — available data range and auto-split recommendation
  - `POST /api/run` — start a training run
  - `POST /api/stop` — stop the running training

## Stop

Press `Ctrl+C` in the terminal to shut down the server. Any running training process will be stopped automatically.
