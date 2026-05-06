"""
Batch Backtest Runner
Tests the parabolic reversal strategy across years of historical setups.
"""
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import csv

from src.backtest.historical_screener import HistoricalParabolicScreener, ParabolicSetup
from src.backtest.tick_backtest_engine_v5 import TickBacktestEngineV5, tick_backtest_engine_v5
from src.backtest.visualizer import visualizer
from src.utils.logger import logger


@dataclass
class BatchBacktestResult:
    """Aggregated results from multiple backtests."""
    total_setups_tested: int = 0
    setups_with_trades: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_pnl: float = 0.0
    avg_pnl_per_setup: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    avg_trade_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # By symbol
    results_by_symbol: Dict[str, List] = field(default_factory=dict)
    
    # Monthly performance
    monthly_pnl: Dict[str, float] = field(default_factory=dict)
    
    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*70)
        print("BATCH BACKTEST RESULTS (Multi-Year)")
        print("="*70)
        print(f"\nSETUP STATISTICS:")
        print(f"  Total Setups Scanned:     {self.total_setups_tested}")
        print(f"  Setups with Trades:       {self.setups_with_trades}")
        conversion = self.setups_with_trades/self.total_setups_tested if self.total_setups_tested > 0 else 0
        print(f"  Conversion Rate:          {conversion:.1%}")
        
        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades:             {self.total_trades}")
        print(f"  Winning Trades:           {self.winning_trades}")
        print(f"  Losing Trades:            {self.losing_trades}")
        print(f"  Win Rate:                 {self.win_rate:.1%}")
        print(f"  Profit Factor:            {self.profit_factor:.2f}")
        
        print(f"\nP&L STATISTICS:")
        print(f"  Total P&L:                ${self.total_pnl:+,.2f}")
        print(f"  Avg P&L per Setup:        ${self.avg_pnl_per_setup:+,.2f}")
        print(f"  Avg Return per Trade:     {self.avg_trade_return:+.2f}%")
        print(f"  Max Drawdown:             ${self.max_drawdown:,.2f}")
        print(f"  Sharpe Ratio:             {self.sharpe_ratio:.2f}")
        
        print(f"\nTOP PERFORMING SYMBOLS:")
        symbol_pnl = {}
        for symbol, results in self.results_by_symbol.items():
            total = sum(r.get('pnl', 0) for r in results)
            symbol_pnl[symbol] = total
        
        top_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)[:10]
        for symbol, pnl in top_symbols:
            print(f"  {symbol:<10} ${pnl:+,.2f}")
        
        print("="*70)


class BatchBacktestRunner:
    """
    Runs backtests across years of historical parabolic setups.
    """
    
    def __init__(self, engine=None):
        self.engine = engine or tick_backtest_engine_v5
        self.screener = HistoricalParabolicScreener()
        self.results: List[Dict] = []
        
    def run_historical_backtest(
        self,
        start_year: int = 2019,
        end_year: int = 2024,
        symbols: Optional[List[str]] = None,
        min_gain_percent: float = 50.0,
        max_setups: Optional[int] = None,
        verbose: bool = False
    ) -> BatchBacktestResult:
        """
        Run comprehensive backtest across years of data.
        
        Parameters:
        -----------
        start_year, end_year : int
            Year range to test (Alpaca has 6+ years of data)
        symbols : List[str], optional
            Universe of symbols (loads default micro-caps if None)
        min_gain_percent : float
            Minimum gain to qualify as parabolic
        max_setups : int, optional
            Limit number of setups to test (for quick testing)
        verbose : bool
            Print detailed output for each setup
        """
        print(f"\n{'='*70}")
        print(f"HISTORICAL BACKTEST: {start_year} - {end_year}")
        print(f"{'='*70}\n")
        
        # Load symbol universe
        if symbols is None:
            symbols = self.screener.load_micro_cap_universe()
        
        # Date range
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        print(f"Scanning {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        print(f"Looking for parabolic moves: {min_gain_percent}%+")
        print()
        
        # Phase 1: Scan for setups
        setups = self.screener.scan_for_parabolic_setups(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            min_gain_percent=min_gain_percent,
            use_cache=True
        )
        
        if not setups:
            print("No parabolic setups found!")
            return BatchBacktestResult()
        
        print(f"\nFound {len(setups)} total parabolic setups")
        
        # Phase 2: Filter to quality setups
        # Support both First Red Day (multi-day) and Intraday Exhaustion (single day)
        quality_setups = self.screener.filter_quality_setups(
            setups,
            min_days_up=1,  # Include single-day parabolic moves
            max_days_up=5,
            min_prior_gain=0.0,  # No prior gain required for intraday
            min_gain_percent=60.0  # Higher threshold for intraday
        )
        
        print(f"Filtered to {len(quality_setups)} quality setups (including intraday)")
        
        # Phase 3: Limit if needed
        if max_setups and len(quality_setups) > max_setups:
            print(f"Limiting to {max_setups} setups for testing")
            quality_setups = quality_setups[:max_setups]
        
        # Phase 4: Run backtest on each setup
        print(f"\nRunning tick-level backtests...")
        print(f"{'Date':<12} {'Symbol':<8} {'Gain':<8} {'Trades':<8} {'P&L':<12} {'Result'}")
        print("-" * 70)
        
        all_results = []
        
        for i, setup in enumerate(quality_setups):
            if i % 10 == 0 and i > 0:
                print(f"\nProgress: {i}/{len(quality_setups)} setups tested")
            
            # Run backtest
            result = self.engine.run_tick_backtest(
                symbol=setup.symbol,
                date=setup.date,
                verbose=False  # Suppress individual output
            )
            
            # Store result
            result_data = {
                'setup': setup,
                'backtest': result,
                'date': setup.date.strftime('%Y-%m-%d'),
                'symbol': setup.symbol,
                'gain_percent': setup.gain_percent,
                'trades': result.total_trades,
                'pnl': result.total_pnl,
                'win_rate': result.win_rate
            }
            all_results.append(result_data)
            
            # Print summary line
            pnl_str = f"${result.total_pnl:+.2f}"
            status = "TRADE" if result.total_trades > 0 else "SKIP"
            print(f"{setup.date.strftime('%Y-%m-%d'):<12} {setup.symbol:<8} "
                  f"{setup.gain_percent:>6.1f}%  {result.total_trades:<8} "
                  f"{pnl_str:<12} {status}")
        
        print(f"\n{'='*70}")
        print("BACKTEST COMPLETE - Generating Report...")
        print(f"{'='*70}")
        
        # Phase 5: Aggregate results
        batch_result = self._aggregate_results(all_results)
        
        # Generate reports
        self._generate_comprehensive_report(batch_result, all_results)
        
        return batch_result
    
    def _aggregate_results(self, all_results: List[Dict]) -> BatchBacktestResult:
        """Aggregate all individual backtest results."""
        total_setups = len(all_results)
        setups_with_trades = sum(1 for r in all_results if r['trades'] > 0)
        
        total_trades = sum(r['trades'] for r in all_results)
        total_pnl = sum(r['pnl'] for r in all_results)
        
        winning_trades = 0
        losing_trades = 0
        total_return = 0.0
        
        for r in all_results:
            result = r['backtest']
            winning_trades += result.winning_trades
            losing_trades += result.losing_trades
            total_return += result.average_trade * result.total_trades
        
        avg_pnl_per_setup = total_pnl / total_setups if total_setups > 0 else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_trade_return = total_return / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        wins = []
        losses = []
        for r in all_results:
            for audit in r['backtest'].audit_records:
                if audit.action.value == 'exit' and audit.pnl is not None:
                    if audit.pnl > 0:
                        wins.append(audit.pnl)
                    else:
                        losses.append(audit.pnl)
        
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Group by symbol
        by_symbol = defaultdict(list)
        for r in all_results:
            by_symbol[r['symbol']].append(r)
        
        # Monthly breakdown
        monthly = defaultdict(float)
        for r in all_results:
            month_key = r['date'][:7]  # YYYY-MM
            monthly[month_key] += r['pnl']
        
        return BatchBacktestResult(
            total_setups_tested=total_setups,
            setups_with_trades=setups_with_trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            avg_pnl_per_setup=avg_pnl_per_setup,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            results_by_symbol=dict(by_symbol),
            monthly_pnl=dict(monthly)
        )
    
    def _generate_comprehensive_report(
        self,
        batch_result: BatchBacktestResult,
        all_results: List[Dict]
    ):
        """Generate comprehensive HTML and CSV reports."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV report
        csv_path = f"reports/batch_backtest_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Date', 'Symbol', 'Day_Gain_%', 'Trades', 'P&L', 
                'Win_Rate', 'Entry_Price', 'Exit_Price', 'Exit_Reason'
            ])
            
            for r in all_results:
                setup = r['setup']
                result = r['backtest']
                
                for audit in result.audit_records:
                    if audit.action.value == 'exit':
                        writer.writerow([
                            setup.date.strftime('%Y-%m-%d'),
                            setup.symbol,
                            f"{setup.gain_percent:.2f}",
                            result.total_trades,
                            f"{audit.pnl:.2f}" if audit.pnl else "0.00",
                            f"{result.win_rate:.2f}",
                            f"{audit.price:.2f}",
                            f"{audit.exit_price:.2f}" if audit.exit_price else "",
                            audit.exit_reason or ""
                        ])
        
        print(f"\nCSV report saved: {csv_path}")
        
        # HTML report
        html_content = self._generate_html_report(batch_result, all_results)
        html_path = f"reports/batch_backtest_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved: {html_path}")
    
    def _generate_html_report(
        self,
        batch_result: BatchBacktestResult,
        all_results: List[Dict]
    ) -> str:
        """Generate comprehensive HTML report."""
        # Simplified HTML - in production this would be more elaborate
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Backtest Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .metric {{ background: white; padding: 20px; border-radius: 8px; 
                  display: inline-block; margin: 10px; min-width: 200px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        table {{ width: 100%; background: white; border-radius: 8px; margin-top: 20px; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .trade-row:hover {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Parabolic Reversal Strategy - Batch Backtest Results</h1>
        <p>{batch_result.total_setups_tested} setups tested | {batch_result.total_trades} trades executed</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value {'positive' if batch_result.total_pnl > 0 else 'negative'}">
                ${batch_result.total_pnl:+,.2f}
            </div>
            <div>Total P&L</div>
        </div>
        <div class="metric">
            <div class="metric-value">{batch_result.win_rate:.1%}</div>
            <div>Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{batch_result.profit_factor:.2f}</div>
            <div>Profit Factor</div>
        </div>
        <div class="metric">
            <div class="metric-value">${batch_result.avg_pnl_per_setup:+,.2f}</div>
            <div>Avg per Setup</div>
        </div>
    </div>
    
    <h2>Trade Log</h2>
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Symbol</th>
                <th>Day Gain</th>
                <th>Trades</th>
                <th>P&L</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for r in all_results[:100]:  # Show first 100
            pnl_class = "positive" if r['pnl'] > 0 else "negative"
            html += f"""
            <tr class="trade-row">
                <td>{r['date']}</td>
                <td>{r['symbol']}</td>
                <td>{r['gain_percent']:.1f}%</td>
                <td>{r['trades']}</td>
                <td class="{pnl_class}">${r['pnl']:+.2f}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
</body>
</html>
"""
        return html


# Singleton
batch_runner = BatchBacktestRunner()
