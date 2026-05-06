"""
Real-time training dashboard server with command center.

Serves the dashboard HTML page, API endpoints for training metrics,
and process management for starting/stopping training from the UI.

Usage:
    python src/dashboard/serve.py --metrics-dir models/wfo_test
    python src/dashboard/serve.py --metrics-dir models/wfo --port 8051
"""

import argparse
import json
import os
import subprocess
import time
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


# =============================================================================
# Process Manager — owns the lifecycle of one training subprocess
# =============================================================================

class ProcessManager:
    """Manages a single training subprocess running in WSL."""

    def __init__(self):
        self.process = None
        self.mode = "idle"       # "idle", "quick", "full"
        self.status = "idle"     # "idle", "running", "completed", "failed", "stopped"
        self.started_at = None
        self.log_path = None
        self._log_fh = None

    def start(self, mode: str, params: dict, metrics_dir: Path) -> dict:
        """Start a training process in WSL."""
        # Reject if already running
        if self.process is not None and self.process.poll() is None:
            return {"ok": False, "error": "Training already running"}

        if mode not in ("custom", "full"):
            return {"ok": False, "error": f"Invalid mode: {mode}"}

        # Clear old dashboard files for a fresh run
        for fname in ("training_metrics.jsonl", "trades.jsonl"):
            fpath = metrics_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text("")

        # Build WSL command
        self.log_path = metrics_dir / "process_output.log"
        cmd = self._build_wsl_command(mode, params, metrics_dir)

        try:
            self._log_fh = open(self.log_path, "w")
            # Log the full command for debugging param forwarding
            self._log_fh.write(f"[CMD] {cmd}\n")
            self._log_fh.write(f"[PARAMS] {json.dumps(params)}\n")
            self._log_fh.flush()
            self.process = subprocess.Popen(
                ["wsl", "bash", "-c", cmd],
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
            )
            self.mode = mode
            self.status = "running"
            self.started_at = datetime.now().isoformat()
            return {"ok": True, "pid": self.process.pid, "mode": mode}
        except Exception as e:
            self.status = "failed"
            return {"ok": False, "error": str(e)}

    def stop(self) -> dict:
        """Stop the running training process."""
        if self.process is None or self.process.poll() is not None:
            return {"ok": False, "error": "No training running"}

        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)

            # Belt and suspenders: kill any lingering Ray/training processes
            try:
                subprocess.run(
                    ["wsl", "bash", "-c", "pkill -f train_wfo; pkill -f ray 2>/dev/null"],
                    timeout=5, capture_output=True
                )
            except Exception:
                pass

            self.status = "stopped"
            self._close_log()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_status(self) -> dict:
        """Poll process state and return status dict."""
        if self.process is not None:
            rc = self.process.poll()
            if rc is not None and self.status == "running":
                self.status = "completed" if rc == 0 else "failed"
                self._close_log()

        elapsed = None
        if self.started_at and self.status == "running":
            elapsed = (datetime.now() - datetime.fromisoformat(self.started_at)).total_seconds()

        return {
            "running": self.status == "running",
            "mode": self.mode,
            "status": self.status,
            "pid": self.process.pid if self.process else None,
            "started_at": self.started_at,
            "elapsed": elapsed,
        }

    def get_log(self, n: int = 100) -> list:
        """Return last n lines of process output."""
        if not self.log_path or not self.log_path.exists():
            return []
        try:
            # Flush if still writing
            if self._log_fh and not self._log_fh.closed:
                self._log_fh.flush()
            lines = self.log_path.read_text(errors="replace").splitlines()
            return lines[-n:]
        except Exception:
            return []

    def _close_log(self):
        if self._log_fh and not self._log_fh.closed:
            self._log_fh.close()
            self._log_fh = None

    def _build_wsl_command(self, mode: str, params: dict, metrics_dir: Path) -> str:
        """Build the full WSL bash command with CLI args from params.

        Modes:
          - "custom": use all params as-is from the UI (user controls everything)
          - "full": auto-compute 70/30 train/test split from available data,
                    override train_months and test_months, keep other params
        """
        # Both modes use the same training script
        script = "src/scripts/train_wfo_quick_test.py"
        wsl_dir = _windows_to_wsl_path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        output_wsl = _windows_to_wsl_path(os.path.abspath(str(metrics_dir)))

        # For "full" mode: auto-compute optimal train/test split and use production defaults
        if mode == "full":
            data_range = _get_data_range_info()
            total_months = data_range.get("total_months", 54)
            purge_days = int(params.get("purge_days", 10))
            # Full mode ALWAYS uses 3 folds for cross-regime robustness
            n_folds = 3
            train_months, test_months = _compute_auto_split(
                total_months, purge_days, n_folds=n_folds
            )
            params["train_months"] = train_months
            params["test_months"] = test_months
            params["n_folds"] = n_folds
            # Full training uses 300K total steps PER FOLD (joint actor-critic)
            if "total_timesteps" not in params or params.get("total_timesteps", 0) < 100000:
                params["total_timesteps"] = 300000
            if "actor_warmup_steps" not in params:
                params["actor_warmup_steps"] = 10000
            if "alpha_start" not in params:
                params["alpha_start"] = 0.2
            if "alpha_end" not in params:
                params["alpha_end"] = 0.01
            # Backwards compat: map to old CLI args
            params["warmup_timesteps"] = 0
            params["finetune_timesteps"] = params["total_timesteps"]
            # Full training uses larger batch and buffer for stability
            if "batch_size" not in params or params.get("batch_size", 0) < 512:
                params["batch_size"] = 512
            if "buffer_size" not in params or params.get("buffer_size", 0) < 200000:
                params["buffer_size"] = 200000
            # Ensure optimized defaults for new params
            if "min_vwap_deviation_entry" not in params:
                params["min_vwap_deviation_entry"] = 15.0
            if "max_drawdown" not in params:
                params["max_drawdown"] = -15000
            if "eval_episodes" not in params:
                params["eval_episodes"] = 50

        # Build CLI args from params (only numeric values for safety)
        cli_parts = [f"--output-dir {output_wsl}"]

        param_map = {
            "total_timesteps": "--total-steps",
            "actor_warmup_steps": "--actor-warmup-steps",
            "alpha_start": "--alpha-start",
            "alpha_end": "--alpha-end",
            "warmup_timesteps": "--warmup-steps",
            "finetune_timesteps": "--finetune-steps",
            "train_months": "--train-months",
            "test_months": "--test-months",
            "purge_days": "--purge-days",
            "batch_size": "--batch-size",
            "buffer_size": "--buffer-size",
            "tau": "--tau",
            "gamma": "--gamma",
            "alpha": "--alpha",
            "lr_actor": "--lr-actor",
            "lr_critic": "--lr-critic",
            "initial_capital": "--initial-capital",
            "max_drawdown": "--max-drawdown",
            "intra_step_stop_loss": "--stop-loss",
            "max_position_capital_fraction": "--max-pos-fraction",
            "min_vwap_deviation_entry": "--vwap-threshold",
            "transaction_cost_per_dollar": "--txn-cost",
            "eval_episodes": "--eval-episodes",
            "n_folds": "--n-folds",
        }

        for key, flag in param_map.items():
            if key in params and params[key] is not None:
                val = params[key]
                # Security: only accept numeric values
                try:
                    float(val)
                except (ValueError, TypeError):
                    continue
                cli_parts.append(f"{flag} {val}")

        cli_args = " ".join(cli_parts)
        return (
            f"cd {wsl_dir} && "
            f"source venv_wsl/bin/activate && "
            f"export PYTHONPATH={wsl_dir}/src:$PYTHONPATH && "
            f"export POLARS_ALLOW_FORKING_THREAD=1 && "
            f"export PYTHONUNBUFFERED=1 && "
            f"python {script} {cli_args}"
        )


def _compute_auto_split(total_months: int, purge_days: int = 10,
                        ratio: float = 0.70, n_folds: int = 3) -> tuple:
    """Compute optimal train/test split from available data.

    For multi-fold WFO, each fold slides forward by test_months.
    Total data needed: train + purge + test + (n_folds - 1) * test.

    With 54 months of data and 3 folds:
      train=24, test=6, purge~0.3 → total needed = 24 + 0.3 + 6 + 2*6 = 42.3 months ✓

    Returns (train_months, test_months).
    """
    purge_months = purge_days / 30.0
    # Total data consumed: 1 fold base + (n_folds-1) sliding test windows
    # train + purge + test + (n_folds - 1) * test = train + purge + n_folds * test
    # Solving for usable data split with given ratio:
    #   usable = total - purge
    #   train + n_folds * test = usable
    #   train / (train + test) = ratio  →  train = ratio * (train + test)
    #   Let S = train + test → train = ratio*S, test = (1-ratio)*S
    #   ratio*S + n_folds * (1-ratio)*S = usable
    #   S * (ratio + n_folds*(1-ratio)) = usable
    usable = total_months - purge_months
    denominator = ratio + n_folds * (1 - ratio)
    s = usable / denominator
    train = int(s * ratio)
    test = int(s * (1 - ratio))
    # Ensure minimums
    train = max(train, 3)
    test = max(test, 1)
    return train, test


def _get_data_range_info() -> dict:
    """Read CSV to get data range info. Used by both the API endpoint and auto-split."""
    import csv as csv_mod
    csv_path = Path("reports/all_setups_backtest.csv")
    if not csv_path.exists():
        return {"min_date": None, "max_date": None, "total_months": 0, "total_setups": 0}
    dates = []
    with open(csv_path, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            d = row.get('Date', '').strip()
            if d:
                dates.append(d)
    if not dates:
        return {"min_date": None, "max_date": None, "total_months": 0, "total_setups": 0}
    min_date = min(dates)
    max_date = max(dates)
    from datetime import datetime as dt
    d0 = dt.strptime(min_date, '%Y-%m-%d')
    d1 = dt.strptime(max_date, '%Y-%m-%d')
    total_months = round((d1 - d0).days / 30.0)
    train, test = _compute_auto_split(total_months, n_folds=3)
    return {
        "min_date": min_date,
        "max_date": max_date,
        "total_months": total_months,
        "total_setups": len(dates),
        "auto_split": {
            "train_months": train, "test_months": test,
            "n_folds": 3, "ratio": "70/30",
        },
    }


def _windows_to_wsl_path(win_path: str) -> str:
    """Convert C:\\foo\\bar to /mnt/c/foo/bar."""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return f"/mnt/{drive}{p[2:]}"
    return p


def _fetch_spy_data(start_date: str, end_date: str, cache_dir: Path) -> dict:
    """Fetch SPY daily close prices. Uses cache, tries yfinance, then Alpaca."""
    cache_path = cache_dir / "spy_cache.json"

    # Check cache
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("start") <= start_date and cached.get("end") >= end_date:
                return {"dates": cached["dates"], "prices": cached["prices"]}
        except Exception:
            pass

    dates_out, prices_out = [], []

    # Try yfinance
    try:
        import yfinance as yf
        from datetime import timedelta as _td
        spy = yf.download("SPY", start=start_date,
                          end=(datetime.strptime(end_date, "%Y-%m-%d") + _td(days=1)).strftime("%Y-%m-%d"),
                          progress=False)
        if spy is not None and len(spy) > 0:
            for idx, row in spy.iterrows():
                d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                close = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
                dates_out.append(d)
                prices_out.append(round(close, 2))
    except Exception:
        pass

    # Try Alpaca if yfinance failed
    if not dates_out:
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_SECRET", "")
            if api_key and api_secret:
                client = StockHistoricalDataClient(api_key, api_secret)
                request = StockBarsRequest(
                    symbol_or_symbols="SPY",
                    timeframe=TimeFrame.Day,
                    start=datetime.strptime(start_date, "%Y-%m-%d"),
                    end=datetime.strptime(end_date, "%Y-%m-%d"),
                )
                bars = client.get_stock_bars(request)
                for bar in bars["SPY"]:
                    dates_out.append(bar.timestamp.strftime("%Y-%m-%d"))
                    prices_out.append(round(float(bar.close), 2))
        except Exception:
            pass

    if not dates_out:
        return None

    # Cache to disk
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "start": dates_out[0],
            "end": dates_out[-1],
            "dates": dates_out,
            "prices": prices_out,
        }))
    except Exception:
        pass

    return {"dates": dates_out, "prices": prices_out}


# =============================================================================
# HTTP Handler
# =============================================================================

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the training dashboard and command center."""

    metrics_dir: Path = Path("models/wfo_test")
    html_path: Path = Path(__file__).parent / "index.html"
    process_manager: ProcessManager = ProcessManager()

    # --- GET ---
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/metrics":
            self._serve_metrics()
        elif path == "/api/trades":
            self._serve_trades()
        elif path == "/api/trades-summary":
            self._serve_trades_summary()
        elif path == "/api/test-results":
            self._serve_test_results()
        elif path == "/api/config":
            self._serve_config()
        elif path == "/api/status":
            self._send_json(self.process_manager.get_status())
        elif path == "/api/process-log":
            self._send_json({"lines": self.process_manager.get_log(150)})
        elif path == "/api/data-range":
            self._serve_data_range()
        elif path == "/api/performance":
            self._serve_performance()
        else:
            self.send_error(404)

    # --- POST ---
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_length > 0:
            try:
                body = json.loads(self.rfile.read(content_length))
            except Exception:
                self._send_json({"ok": False, "error": "Invalid JSON"})
                return

        path = self.path.split("?")[0]
        if path == "/api/run":
            mode = body.get("mode", "quick")
            params = body.get("params", {})
            result = self.process_manager.start(mode, params, self.metrics_dir)
            self._send_json(result)
        elif path == "/api/stop":
            result = self.process_manager.stop()
            self._send_json(result)
        else:
            self.send_error(404)

    # --- OPTIONS (CORS preflight) ---
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # --- Existing GET handlers ---
    def _serve_html(self):
        try:
            content = self.html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(500, "index.html not found")

    def _serve_metrics(self):
        self._send_json(self._read_jsonl("training_metrics.jsonl"))

    def _serve_trades(self):
        trades = self._read_jsonl("trades.jsonl")
        self._send_json(trades[-5000:][::-1])

    def _serve_trades_summary(self):
        trades = self._read_jsonl("trades.jsonl")
        if not trades:
            self._send_json(None)
            return

        def _summarize(trade_list):
            if not trade_list:
                return None
            pnls = [t.get("pnl", 0) for t in trade_list]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            bars = [t.get("bars_held", 0) for t in trade_list]
            mfes = [t.get("mfe", 0) for t in trade_list]
            maes = [t.get("mae", 0) for t in trade_list]
            gross_profit = sum(wins)
            gross_loss = sum(losses)
            return {
                "total_trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
                "win_count": len(wins),
                "loss_count": len(losses),
                "win_rate": round(len(wins) / len(pnls), 3) if pnls else 0,
                "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
                "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
                "largest_win": round(max(pnls), 2) if pnls else 0,
                "largest_loss": round(min(pnls), 2) if pnls else 0,
                "profit_factor": round(gross_profit / abs(gross_loss), 2) if gross_loss != 0 else 0,
                "avg_bars_held": round(sum(bars) / len(bars), 1) if bars else 0,
                "avg_mfe": round(sum(mfes) / len(mfes), 2) if mfes else 0,
                "avg_mae": round(sum(maes) / len(maes), 2) if maes else 0,
            }

        warmup = [t for t in trades if t.get("phase", "unknown") in ("warmup_actor_frozen", "unknown")]
        finetuning = [t for t in trades if t.get("phase", "").startswith("finetuning")]
        evaluation = [t for t in trades if t.get("phase", "") == "evaluation"]
        result = _summarize(trades)
        result["by_phase"] = {
            "warmup": _summarize(warmup),
            "finetuning": _summarize(finetuning),
            "evaluation": _summarize(evaluation),
        }
        self._send_json(result)

    def _serve_test_results(self):
        for name in ["quick_test_results.json", "wfo_results.json"]:
            path = self.metrics_dir / name
            if path.exists():
                try:
                    self._send_json(json.loads(path.read_text()))
                    return
                except Exception:
                    pass
        self._send_json(None)

    def _serve_config(self):
        result = {"config": None, "folds": [], "active_fold": None}
        metrics = self._read_jsonl("training_metrics.jsonl")
        if metrics:
            result["active_fold"] = metrics[-1].get("fold")
        for name in ["quick_test_results.json", "wfo_results.json"]:
            path = self.metrics_dir / name
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    result["config"] = data.get("config")
                    for fold in data.get("folds", []):
                        slim = {k: v for k, v in fold.items() if k != "per_episode_results"}
                        result["folds"].append(slim)
                    break
                except Exception:
                    pass
        self._send_json(result)

    # --- Helpers ---
    def _read_jsonl(self, filename: str) -> list:
        path = self.metrics_dir / filename
        results = []
        if path.exists():
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        return results

    def _serve_performance(self):
        """Portfolio vs SPY performance comparison from OOS evaluation results."""
        import math as _math

        # 1. Read test results
        results_data = None
        for name in ["quick_test_results.json", "wfo_results.json"]:
            path = self.metrics_dir / name
            if path.exists():
                try:
                    results_data = json.loads(path.read_text())
                    break
                except Exception:
                    pass
        if not results_data or "folds" not in results_data:
            self._send_json(None)
            return

        # 2. Extract per-episode results across all folds, sorted by date
        episodes = []
        for fold in results_data.get("folds", []):
            for ep in fold.get("per_episode_results", []):
                if ep.get("date"):
                    episodes.append(ep)
        episodes.sort(key=lambda e: e["date"])
        if not episodes:
            self._send_json(None)
            return

        initial_capital = 100000.0
        dates = [ep["date"] for ep in episodes]
        pnls = [ep.get("pnl", 0.0) for ep in episodes]
        trades_per_ep = [ep.get("trades", 0) for ep in episodes]

        # 3. Build portfolio equity curve
        equity = []
        cumulative = initial_capital
        for pnl in pnls:
            cumulative += pnl
            equity.append(round(cumulative, 2))

        cum_return_pct = [round((eq / initial_capital - 1) * 100, 4) for eq in equity]

        # 4. Fetch SPY data
        spy_data = _fetch_spy_data(dates[0], dates[-1], self.metrics_dir)

        # 5. Compute portfolio metrics
        total_pnl = sum(pnls)
        total_return_pct = (total_pnl / initial_capital) * 100
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Daily returns for risk metrics
        daily_returns = [p / initial_capital for p in pnls]
        n = len(daily_returns)
        mean_r = sum(daily_returns) / n if n else 0
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / max(n - 1, 1)
        std_r = variance ** 0.5
        downside_var = sum(min(0, r) ** 2 for r in daily_returns) / max(n - 1, 1)
        downside_dev = downside_var ** 0.5

        sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0
        sortino = (mean_r / downside_dev * (252 ** 0.5)) if downside_dev > 0 else 0

        # Max drawdown
        peak = initial_capital
        max_dd = 0
        dd_series = []
        cum_eq = initial_capital
        for pnl in pnls:
            cum_eq += pnl
            if cum_eq > peak:
                peak = cum_eq
            dd = cum_eq - peak
            if dd < max_dd:
                max_dd = dd
            dd_series.append(round(dd, 2))

        # CAGR
        day_span = 1
        try:
            from datetime import datetime as _dt
            d0 = _dt.strptime(dates[0], "%Y-%m-%d")
            d1 = _dt.strptime(dates[-1], "%Y-%m-%d")
            day_span = max((d1 - d0).days, 1)
        except Exception:
            pass
        years = day_span / 365.25
        final_eq = initial_capital + total_pnl
        cagr = ((final_eq / initial_capital) ** (1 / years) - 1) if years > 0 and final_eq > 0 else 0
        calmar = cagr / abs(max_dd / initial_capital) if max_dd < 0 else 0

        portfolio_metrics = {
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_dd": round(max_dd, 2),
            "max_dd_pct": round(max_dd / initial_capital * 100, 2),
            "calmar": round(calmar, 2),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_pnl": round(total_pnl, 2),
            "total_trades": sum(trades_per_ep),
            "episodes": n,
        }

        # 6. SPY metrics + comparison
        spy_response = None
        comparison = None
        spy_metrics = None
        if spy_data:
            spy_prices = spy_data["prices"]
            spy_dates = spy_data["dates"]
            # Normalize SPY to same initial capital
            if spy_prices:
                spy_base = spy_prices[0]
                spy_equity = [round(initial_capital * p / spy_base, 2) for p in spy_prices]
                spy_cum_ret = [round((p / spy_base - 1) * 100, 4) for p in spy_prices]
                spy_total_ret = (spy_prices[-1] / spy_base - 1) * 100
                spy_cagr = ((spy_prices[-1] / spy_base) ** (1 / max(years, 0.01)) - 1) if spy_prices[-1] > 0 else 0

                # SPY drawdown
                spy_peak = spy_prices[0]
                spy_max_dd_pct = 0
                for p in spy_prices:
                    if p > spy_peak:
                        spy_peak = p
                    dd_pct = (p / spy_peak - 1) * 100
                    if dd_pct < spy_max_dd_pct:
                        spy_max_dd_pct = dd_pct

                # SPY daily returns for Sharpe
                spy_daily_ret = []
                for i in range(1, len(spy_prices)):
                    spy_daily_ret.append(spy_prices[i] / spy_prices[i - 1] - 1)
                spy_mean = sum(spy_daily_ret) / len(spy_daily_ret) if spy_daily_ret else 0
                spy_std = (sum((r - spy_mean) ** 2 for r in spy_daily_ret) / max(len(spy_daily_ret) - 1, 1)) ** 0.5
                spy_sharpe = (spy_mean / spy_std * (252 ** 0.5)) if spy_std > 0 else 0

                spy_metrics = {
                    "cagr": round(spy_cagr, 4),
                    "sharpe": round(spy_sharpe, 2),
                    "total_return_pct": round(spy_total_ret, 2),
                    "max_dd_pct": round(spy_max_dd_pct, 2),
                }

                spy_response = {
                    "dates": spy_dates,
                    "equity": spy_equity,
                    "cumulative_return_pct": spy_cum_ret,
                }

                # Alpha and Beta (CAPM)
                # Match portfolio dates to SPY dates for correlation
                spy_date_price = dict(zip(spy_dates, spy_prices))
                matched_port_ret = []
                matched_spy_ret = []
                prev_spy = None
                for i, d in enumerate(dates):
                    if d in spy_date_price:
                        spy_p = spy_date_price[d]
                        if prev_spy is not None and prev_spy > 0:
                            matched_port_ret.append(pnls[i] / initial_capital)
                            matched_spy_ret.append(spy_p / prev_spy - 1)
                        prev_spy = spy_p

                if len(matched_port_ret) > 2:
                    mp = sum(matched_port_ret) / len(matched_port_ret)
                    ms = sum(matched_spy_ret) / len(matched_spy_ret)
                    cov_ps = sum((matched_port_ret[i] - mp) * (matched_spy_ret[i] - ms)
                                 for i in range(len(matched_port_ret))) / (len(matched_port_ret) - 1)
                    var_s = sum((r - ms) ** 2 for r in matched_spy_ret) / (len(matched_spy_ret) - 1)
                    beta = cov_ps / var_s if var_s > 0 else 0
                    alpha_ann = (mp - beta * ms) * 252
                    # Correlation
                    std_p = (sum((r - mp) ** 2 for r in matched_port_ret) / (len(matched_port_ret) - 1)) ** 0.5
                    std_s = var_s ** 0.5
                    corr = cov_ps / (std_p * std_s) if std_p > 0 and std_s > 0 else 0
                    comparison = {
                        "alpha": round(alpha_ann, 4),
                        "beta": round(beta, 3),
                        "correlation": round(corr, 3),
                    }

        # 7. Year-over-year breakdown
        yearly = {}
        for i, ep in enumerate(episodes):
            year = ep["date"][:4]
            if year not in yearly:
                yearly[year] = {"pnl": 0, "trades": 0, "wins": 0, "episodes": 0}
            yearly[year]["pnl"] += ep.get("pnl", 0)
            yearly[year]["trades"] += ep.get("trades", 0)
            yearly[year]["episodes"] += 1
            if ep.get("pnl", 0) > 0:
                yearly[year]["wins"] += 1

        yearly_list = []
        for year in sorted(yearly.keys()):
            y = yearly[year]
            entry = {
                "year": int(year),
                "portfolio_pnl": round(y["pnl"], 2),
                "trades": y["trades"],
                "episodes": y["episodes"],
                "win_rate": round(y["wins"] / y["episodes"], 3) if y["episodes"] > 0 else 0,
            }
            # Add SPY return for this year if available
            if spy_data:
                spy_date_price = dict(zip(spy_data["dates"], spy_data["prices"]))
                year_prices = [(d, p) for d, p in zip(spy_data["dates"], spy_data["prices"])
                               if d.startswith(year)]
                if len(year_prices) >= 2:
                    entry["spy_return_pct"] = round(
                        (year_prices[-1][1] / year_prices[0][1] - 1) * 100, 2
                    )
            yearly_list.append(entry)

        self._send_json({
            "portfolio": {
                "dates": dates,
                "equity": equity,
                "cumulative_return_pct": cum_return_pct,
                "drawdown": dd_series,
            },
            "spy": spy_response,
            "metrics": {
                "portfolio": portfolio_metrics,
                "spy": spy_metrics,
            },
            "comparison": comparison,
            "yearly": yearly_list,
            "initial_capital": initial_capital,
        })

    def _serve_data_range(self):
        """Return data range info including auto-split recommendation."""
        self._send_json(_get_data_range_info())

    def _send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="WFO Training Dashboard & Command Center")
    parser.add_argument("--metrics-dir", type=str, default="models/wfo_test",
                        help="Directory containing training_metrics.jsonl")
    parser.add_argument("--port", type=int, default=8050, help="Server port")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    DashboardHandler.metrics_dir = Path(args.metrics_dir)
    DashboardHandler.html_path = Path(__file__).parent / "index.html"

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"

    print(f"Dashboard: {url}")
    print(f"Metrics:   {DashboardHandler.metrics_dir}")
    print(f"Press Ctrl+C to stop\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        # Clean up training process if running
        pm = DashboardHandler.process_manager
        if pm.process and pm.process.poll() is None:
            print("Stopping training process...")
            pm.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
