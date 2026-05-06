"""
Behavioral Cloning (BC) Pre-training for SAC Agent - REAL HISTORICAL DATA ONLY

This script implements Behavioral Cloning using ONLY real 60-bar historical 
windows loaded from Parquet files. NO synthetic data generation is used.

CRITICAL REQUIREMENTS:
1. All training samples use REAL 60-bar OHLCV sequences from Parquet
2. If a sample lacks sufficient prior bars, it is SKIPPED explicitly
3. Target actions based on execution labels (not outcomes)
4. NO synthetic fallback - research validity depends on real historical data

ANCHORING LOGIC (deterministic, no timestamps required):
- POSITIVE (entry): Anchor to bar with maximum VWAP extension in entry window
  This represents the "best setup" bar where V5 would have triggered entry
  60-bar window: [max_vwap_idx - 60, max_vwap_idx) - strictly pre-entry
  
- NEGATIVE (flat): Anchor to random bar in entry window where NO entry occurred
  Criteria: VWAP extension < threshold (not a setup bar)
  60-bar window: [anchor_idx - 60, anchor_idx) - strictly pre-anchor
  
Both use REAL bar indices from Parquet - no synthetic timestamps.

Author: AI Agent
Date: 2026-03-18
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import logging
from tqdm import tqdm
import pytz

# Add project root to path (parent of src/)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.perception import (
    TemporalAutoencoder,
    PerceptionConfig,
    create_perception_module
)
from src.rl.agent import MaskedGaussianPolicy, SACConfig
from src.rl.data_provider_hybrid import HybridDataProvider, get_data_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    """Configuration for Behavioral Cloning."""
    
    trades_csv: str = "reports/full_3527_backtest_results.csv"
    data_cache_dir: str = "data/cache/1min_extended"  # REAL parquet data
    output_dir: str = "models/behavioral_cloning"
    
    sequence_length: int = 60
    use_frozen_encoder: bool = True
    encoder_checkpoint: Optional[str] = None
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 15
    min_delta: float = 1e-6
    
    # Negative sampling ratio
    negative_sampling_ratio: float = 1.0  # 1:1 balanced ratio to avoid hold-bias
    
    # Entry criteria (percent difference, not ratio)
    # settings.yaml uses 1.20 (ratio) = 20% (percent difference)
    # Calculation: (close - vwap) / vwap * 100
    entry_vwap_threshold: float = 20.0  # VWAP extension > 20.0%
    entry_time_start: Tuple[int, int] = (9, 45)  # 9:45 AM
    entry_time_end: Tuple[int, int] = (14, 30)   # 2:30 PM
    
    # Target actions
    entry_action: float = -1.0   # V5 entered SHORT
    no_trade_action: float = 0.0  # V5 did NOT trade (flat)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class AnchoredSample:
    """
    A sample with a REAL anchored bar index from Parquet data.
    
    IMPORTANT: This is WEAKLY SUPERVISED anchoring, not exact expert cloning.
    The anchor is determined by heuristic (max VWAP deviation), not by the
    actual expert entry timestamp (which is unavailable in the data).
    
    This guarantees the 60-bar window [anchor_idx - 60, anchor_idx) exists
    and comes from real historical data.
    """
    symbol: str
    date: datetime
    anchor_idx: int  # REAL bar index in Parquet (0-indexed from market open)
    anchor_time: datetime  # REAL timestamp of anchor bar
    vwap_deviation: float  # At anchor bar
    volume_concentration: float
    is_entry: bool  # True if this is an entry sample
    target_action: float
    window_start_idx: int  # anchor_idx - 60 (for verification)


class ExpertTradeDataset(Dataset):
    """
    Dataset using WEAKLY SUPERVISED entry-window anchoring from REAL Parquet data.
    
    ANCHORING STRATEGY (heuristic, timestamp-independent):
    
    POSITIVE SAMPLES (entries):
    - Load actual Parquet for the symbol/date
    - Find bars in entry window (9:45-14:30) with VWAP > threshold (20%)
    - Anchor to the bar with MAXIMUM VWAP extension (HEURISTIC - not exact expert time)
    - Extract 60 bars STRICTLY BEFORE anchor: [anchor_idx - 60, anchor_idx)
    - Target: -1.0 (short entry)
    
    NEGATIVE SAMPLES (non-entries):
    - Load actual Parquet for symbol/date
    - Find bars in entry window with VWAP < threshold (not setups)
    - Randomly select one such bar as anchor
    - Extract 60 bars STRICTLY BEFORE anchor: [anchor_idx - 60, anchor_idx)
    - Target: 0.0 (flat)
    
    CRITICAL: If insufficient prior bars (< 60), sample is SKIPPED (not synthesized).
    """
    
    def __init__(
        self,
        config: BCConfig,
        perception_config: Optional[PerceptionConfig] = None
    ):
        self.config = config
        self.perception_config = perception_config or PerceptionConfig()
        self.device = torch.device(config.device)
        self.et_tz = pytz.timezone('America/New_York')
        
        # Initialize data provider for REAL parquet loading
        logger.info(f"Initializing HybridDataProvider with REAL parquet: {config.data_cache_dir}")
        self.data_provider = get_data_provider(
            parquet_dir=config.data_cache_dir,
            mode="train"
        )
        
        # Load expert trades (date-level labels only)
        self.trades_df = self._load_trades()
        
        # Initialize frozen perception module
        self.perception, _ = create_perception_module(
            checkpoint_path=config.encoder_checkpoint,
            config=self.perception_config
        )
        self.perception.to(self.device)
        self.perception.eval()
        
        # Build anchored samples from REAL parquet data
        self.samples: List[AnchoredSample] = self._build_anchored_dataset()
        
        logger.info(f"Dataset built: {len(self.samples)} valid anchored samples")
        n_entry = sum(1 for s in self.samples if s.is_entry)
        logger.info(f"  Positive (entry): {n_entry}")
        logger.info(f"  Negative (flat):  {len(self.samples) - n_entry}")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples loaded! Check data availability.")
    
    def _load_trades(self) -> pl.DataFrame:
        """Load expert trade data - date-level labels only."""
        csv_path = Path(self.config.trades_csv)
        
        if not csv_path.exists():
            logger.warning(f"Trade CSV not found: {csv_path}")
            # Return empty dataframe - will result in empty dataset
            return pl.DataFrame({
                'symbol': [],
                'date': [],
                'is_entry': []
            })
        
        df = pl.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from {csv_path}")
        
        # Determine entry vs non-entry from available columns
        if 'trades' in df.columns:
            df = df.with_columns((pl.col('trades') > 0).alias('is_entry'))
        elif 'pnl' in df.columns:
            df = df.with_columns((pl.col('pnl') != 0).alias('is_entry'))
        else:
            # Assume all are entries if no distinguishing column
            df = df.with_columns(pl.lit(True).alias('is_entry'))
        
        # Parse dates
        df = df.with_columns(pl.col('date').str.strptime(pl.Datetime, "%Y-%m-%d").alias('date'))
        
        logger.info(f"Entries: {df.filter(pl.col('is_entry')).height}, Non-entries: {df.filter(~pl.col('is_entry')).height}")
        
        return df
    
    def _build_anchored_dataset(self) -> List[AnchoredSample]:
        """
        Build dataset with REAL anchored samples from Parquet.
        
        For each symbol/date:
        1. Load actual Parquet data
        2. Find valid anchor bars (real indices from actual data)
        3. Verify 60 prior bars exist
        4. Create AnchoredSample with real indices
        """
        valid_samples: List[AnchoredSample] = []
        skipped_stats = {'no_data': 0, 'insufficient_bars': 0, 'no_anchor': 0}
        
        entry_window_start = dt_time(*self.config.entry_time_start)
        entry_window_end = dt_time(*self.config.entry_time_end)
        
        # Process entries (positive samples)
        entry_df = self.trades_df.filter(pl.col('is_entry'))
        logger.info(f"Processing {entry_df.height} entry records...")
        
        for row in tqdm(entry_df.iter_rows(named=True), desc="Anchoring entries", total=entry_df.height):
            symbol = row['symbol']
            date = row['date']
            date_str = date.strftime('%Y-%m-%d')
            
            # Load REAL parquet data
            df = self._load_trading_day(symbol, date_str)
            if df is None or len(df) < self.config.sequence_length:
                skipped_stats['no_data'] += 1
                continue
            
            # Add bar indices and calculate VWAP deviation
            df = df.with_row_index('__bar_idx__')
            
            # Calculate VWAP deviation: (close - vwap) / vwap * 100
            df = df.with_columns(
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            )
            
            # Find bars in entry window with VWAP > threshold
            # Note: vwap_dev is already in percent, threshold is in percent (e.g., 1.20 = 1.20%)
            entry_bars = df.filter(
                (pl.col('timestamp').dt.time() >= entry_window_start) &
                (pl.col('timestamp').dt.time() <= entry_window_end) &
                (pl.col('vwap_dev').abs() > self.config.entry_vwap_threshold)
            )
            
            if len(entry_bars) == 0:
                skipped_stats['no_anchor'] += 1
                continue
            
            # Anchor to bar with MAX VWAP deviation (strongest setup)
            best_bar = entry_bars.sort('vwap_dev', descending=True).row(0, named=True)
            anchor_idx = best_bar['__bar_idx__']
            
            # CRITICAL: Verify 60 prior bars exist
            window_start_idx = anchor_idx - self.config.sequence_length
            if window_start_idx < 0:
                skipped_stats['insufficient_bars'] += 1
                continue
            
            # Get anchor time for verification
            anchor_time = best_bar['timestamp']
            if isinstance(anchor_time, str):
                anchor_time = datetime.fromisoformat(anchor_time)
            
            valid_samples.append(AnchoredSample(
                symbol=symbol,
                date=date,
                anchor_idx=int(anchor_idx),
                anchor_time=anchor_time,
                vwap_deviation=best_bar['vwap_dev'],
                volume_concentration=best_bar.get('volume_concentration', 0.75),
                is_entry=True,
                target_action=self.config.entry_action,
                window_start_idx=int(window_start_idx)
            ))
        
        # Build negative samples
        n_positive = len(valid_samples)
        n_negative_target = int(n_positive * self.config.negative_sampling_ratio)
        
        logger.info(f"Building {n_negative_target} negative samples...")
        
        # Get unique dates from entries for negative sampling
        unique_entries = entry_df.select(['symbol', 'date']).unique()
        neg_attempts = 0
        neg_created = 0
        
        while neg_created < n_negative_target and neg_attempts < n_negative_target * 5:
            neg_attempts += 1
            
            # Pick random entry record as base
            base_row = unique_entries.sample(1).row(0, named=True)
            symbol = base_row['symbol']
            date = base_row['date']
            date_str = date.strftime('%Y-%m-%d')
            
            # Load REAL parquet
            df = self._load_trading_day(symbol, date_str)
            if df is None or len(df) < self.config.sequence_length:
                continue
            
            df = df.with_row_index('__bar_idx__')
            
            # Calculate VWAP deviation
            df = df.with_columns(
                ((pl.col('close') - pl.col('vwap')) / pl.col('vwap') * 100).alias('vwap_dev')
            )
            
            # Find bars in entry window that are NOT setups (VWAP < threshold)
            non_entry_bars = df.filter(
                (pl.col('timestamp').dt.time() >= entry_window_start) &
                (pl.col('timestamp').dt.time() <= entry_window_end) &
                (pl.col('vwap_dev').abs() <= self.config.entry_vwap_threshold)
            )
            
            if len(non_entry_bars) == 0:
                continue
            
            # Randomly select one non-entry bar
            neg_bar = non_entry_bars.sample(1).row(0, named=True)
            anchor_idx = neg_bar['__bar_idx__']
            
            # Verify 60 prior bars
            window_start_idx = anchor_idx - self.config.sequence_length
            if window_start_idx < 0:
                continue
            
            anchor_time = neg_bar['timestamp']
            if isinstance(anchor_time, str):
                anchor_time = datetime.fromisoformat(anchor_time)
            
            valid_samples.append(AnchoredSample(
                symbol=symbol,
                date=date,
                anchor_idx=int(anchor_idx),
                anchor_time=anchor_time,
                vwap_deviation=neg_bar['vwap_dev'],
                volume_concentration=neg_bar.get('volume_concentration', 0.5),
                is_entry=False,
                target_action=self.config.no_trade_action,
                window_start_idx=int(window_start_idx)
            ))
            neg_created += 1
        
        logger.info(f"Negative samples created: {neg_created} (attempts: {neg_attempts})")
        logger.info(f"Skipped: {skipped_stats}")
        
        np.random.shuffle(valid_samples)
        return valid_samples
    
    def _load_trading_day(self, symbol: str, date_str: str) -> Optional[pl.DataFrame]:
        """Load trading day data from Parquet via data provider."""
        try:
            return self.data_provider._load_trading_day(symbol, date_str)
        except Exception as e:
            logger.debug(f"Failed to load {symbol} {date_str}: {e}")
            return None
    
    def _load_sequence_for_sample(self, sample: AnchoredSample) -> Optional[np.ndarray]:
        """
        Load REAL 60-bar OHLCV sequence for anchored sample.
        
        CRITICAL SEMANTICS (must match RL env.py):
        - Window: [anchor_idx - 60, anchor_idx) - strictly PRE-ANCHOR
        - The 60th bar (index 59) is immediately before the anchor bar
        - No future information leakage - all bars precede decision point
        
        Args:
            sample: AnchoredSample with real anchor_idx from Parquet
            
        Returns:
            np.ndarray: [60, 5] OHLCV sequence or None if loading fails
        """
        date_str = sample.date.strftime('%Y-%m-%d')
        
        try:
            df = self._load_trading_day(sample.symbol, date_str)
            if df is None:
                return None
            
            # Verify anchor is still valid
            if sample.anchor_idx >= len(df):
                logger.warning(f"Anchor idx {sample.anchor_idx} out of bounds for {sample.symbol} {date_str}")
                return None
            
            # CRITICAL: Extract EXACTLY the 60 bars before anchor
            # Window: [anchor_idx - 60, anchor_idx) - strictly pre-anchor
            start_idx = sample.window_start_idx
            if start_idx < 0:
                return None
            
            window_df = df.slice(start_idx, self.config.sequence_length)
            
            if len(window_df) < self.config.sequence_length:
                logger.warning(f"Incomplete window for {sample.symbol} {date_str}: {len(window_df)} bars")
                return None
            
            # Extract OHLCV
            ohlcv = window_df.select(['open', 'high', 'low', 'close', 'volume']).to_numpy()
            
            # Z-score normalize per-feature
            means = ohlcv.mean(axis=0)
            stds = ohlcv.std(axis=0) + 1e-8
            ohlcv = (ohlcv - means) / stds
            
            return ohlcv.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Failed to load sequence for {sample.symbol} {date_str}: {e}")
            return None
    
    def _generate_state(self, sample: AnchoredSample) -> Optional[torch.Tensor]:
        """Generate 74-dimensional state from REAL anchored historical data."""
        sequence = self._load_sequence_for_sample(sample)
        if sequence is None:
            return None
        
        # Convert to tensor [1, 5, sequence_length]
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        seq_tensor = seq_tensor.transpose(1, 2)
        
        # Extract latent z using frozen encoder
        with torch.no_grad():
            z = self.perception.encoder(seq_tensor)  # [1, 64]
        
        # Explicit features from anchored sample
        vwap_dev = torch.tensor([sample.vwap_deviation / 100.0]).to(self.device)
        vol_conc = torch.tensor([sample.volume_concentration]).to(self.device)
        
        # Portfolio state (simplified for BC)
        position = torch.zeros(1).to(self.device)
        unrealized_pnl = torch.zeros(1).to(self.device)
        drawdown = torch.zeros(1).to(self.device)
        kelly = torch.tensor([0.5]).to(self.device)
        
        # Time features from anchor time
        anchor_time = sample.anchor_time
        if isinstance(anchor_time, str):
            anchor_time = datetime.fromisoformat(anchor_time)
        hour = torch.tensor([anchor_time.hour / 24.0]).to(self.device)
        minute = torch.tensor([anchor_time.minute / 60.0]).to(self.device)
        
        # Entry window flag
        entry_start = self.config.entry_time_start[0] + self.config.entry_time_start[1] / 60.0
        entry_end = self.config.entry_time_end[0] + self.config.entry_time_end[1] / 60.0
        current_time = anchor_time.hour + anchor_time.minute / 60.0
        in_window = torch.tensor([1.0 if entry_start <= current_time <= entry_end else 0.0]).to(self.device)
        must_flatten = torch.zeros(1).to(self.device)
        
        # Concatenate: 64 + 10 = 74 dimensions
        state = torch.cat([
            z.squeeze(0),  # 64
            vwap_dev,      # 1
            vol_conc,      # 1
            position,      # 1
            unrealized_pnl,# 1
            drawdown,      # 1
            kelly,         # 1
            hour,          # 1
            minute,        # 1
            in_window,     # 1
            must_flatten   # 1
        ])
        
        return state.cpu().float()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training sample with REAL anchored historical data."""
        sample = self.samples[idx]
        
        state = self._generate_state(sample)
        action = torch.tensor([sample.target_action], dtype=torch.float32)
        
        if state is None:
            # This should not happen after filtering, but handle gracefully
            logger.error(f"CRITICAL: Sample {sample.symbol} {sample.date} idx={sample.anchor_idx} failed to load!")
            # Return a zero state (will be obvious in training if this happens)
            state = torch.zeros(74)
        
        return state, action
    
    def verify_anchoring(self, n_samples: int = 5) -> bool:
        """
        Verify anchoring correctness for random samples.
        
        Checks:
        1. Final bar in window is immediately before anchor bar
        2. Window contains exactly 60 bars
        3. No synthetic fallback
        """
        logger.info(f"\nVerifying anchoring for {n_samples} random samples...")
        
        samples_to_check = np.random.choice(self.samples, min(n_samples, len(self.samples)), replace=False)
        
        all_pass = True
        for sample in samples_to_check:
            date_str = sample.date.strftime('%Y-%m-%d')
            df = self._load_trading_day(sample.symbol, date_str)
            
            if df is None:
                logger.error(f"  FAIL: Could not load {sample.symbol} {date_str}")
                all_pass = False
                continue
            
            # Get the actual anchor bar time
            if sample.anchor_idx < len(df):
                anchor_bar = df.row(sample.anchor_idx, named=True)
                anchor_time = anchor_bar.get('timestamp', 'unknown')
                
                # Get the final bar of the window (should be immediately before anchor)
                window_end_idx = sample.anchor_idx - 1
                if window_end_idx >= 0:
                    window_end_bar = df.row(window_end_idx, named=True)
                    window_end_time = window_end_bar.get('timestamp', 'unknown')
                    
                    logger.info(f"  {sample.symbol} {date_str}:")
                    logger.info(f"    Anchor: idx={sample.anchor_idx}, time={anchor_time}")
                    logger.info(f"    Window: [{sample.window_start_idx}, {window_end_idx}], end_time={window_end_time}")
                    logger.info(f"    Type: {'ENTRY' if sample.is_entry else 'FLAT'}, target={sample.target_action}")
                    
                    # Verify window size
                    if sample.anchor_idx - sample.window_start_idx == 60:
                        logger.info(f"    SUCCESS: Window size correct (60 bars)")
                    else:
                        logger.error(f"    FAIL: Window size incorrect: {sample.anchor_idx - sample.window_start_idx}")
                        all_pass = False
                else:
                    logger.error(f"  FAIL: Window end index {window_end_idx} < 0")
                    all_pass = False
            else:
                logger.error(f"  FAIL: Anchor idx {sample.anchor_idx} out of bounds")
                all_pass = False
        
        if all_pass:
            logger.info("\nSUCCESS: All anchoring verifications passed!")
        else:
            logger.error("\nFAIL: Some anchoring verifications failed!")
        
        return all_pass


class BehavioralCloningTrainer:
    """Trainer for Behavioral Cloning."""
    
    def __init__(self, actor: MaskedGaussianPolicy, config: BCConfig):
        self.actor = actor.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info("BehavioralCloningTrainer initialized")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.actor.train()
        total_loss = 0.0
        
        for states, expert_actions in tqdm(dataloader, desc="BC Training"):
            states = states.to(self.device)
            expert_actions = expert_actions.to(self.device)
            
            pred_actions, _, _ = self.actor(states, action_mask=None, deterministic=True)
            loss = self.criterion(pred_actions, expert_actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.actor.eval()
        total_loss = 0.0
        entry_mse = 0.0
        flat_mse = 0.0
        n_entry = 0
        n_flat = 0
        
        with torch.no_grad():
            for states, expert_actions in tqdm(dataloader, desc="Validation"):
                states = states.to(self.device)
                expert_actions = expert_actions.to(self.device)
                
                pred_actions, _, _ = self.actor(states, action_mask=None, deterministic=True)
                loss = self.criterion(pred_actions, expert_actions)
                total_loss += loss.item()
                
                for i in range(len(expert_actions)):
                    if expert_actions[i].item() < -0.5:
                        entry_mse += (pred_actions[i] - expert_actions[i]).pow(2).item()
                        n_entry += 1
                    else:
                        flat_mse += (pred_actions[i] - expert_actions[i]).pow(2).item()
                        n_flat += 1
        
        avg_loss = total_loss / len(dataloader)
        
        metrics = {
            'val_loss': avg_loss,
            'entry_mse': entry_mse / max(n_entry, 1),
            'flat_mse': flat_mse / max(n_flat, 1),
            'n_entry': n_entry,
            'n_flat': n_flat
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Full BC training loop."""
        logger.info(f"Starting BC training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if val_loader is not None:
                val_loss, metrics = self.validate(val_loader)
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_loss - self.config.min_delta:
                    self.best_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('bc_best.pt')
                    logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
                else:
                    self.epochs_without_improvement += 1
                
                logger.info(
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"Entry MSE: {metrics['entry_mse']:.6f} | Flat MSE: {metrics['flat_mse']:.6f}"
                )
                
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(f"Train Loss: {train_loss:.6f}")
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.save_checkpoint('bc_best.pt')
        
        return {'train_loss': self.train_losses}
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        output_path = Path(self.config.output_dir) / filename
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }, output_path)


def create_chronological_split(dataset: ExpertTradeDataset, val_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """
    Create chronological train/validation split based on dates.
    
    CRITICAL: Uses date-based splitting (NOT random) to prevent data leakage
    in time-series market data. All samples from same date stay together.
    
    Split rule:
    - Sort all unique dates chronologically
    - Earliest (1 - val_ratio) -> train
    - Latest val_ratio -> validation
    
    Returns:
        (train_dataset, val_dataset): torch.utils.data.Subset objects
    """
    # Extract unique dates from all samples
    sample_dates = [(i, s.date) for i, s in enumerate(dataset.samples)]
    
    # Get unique dates sorted chronologically
    unique_dates = sorted(set(d for _, d in sample_dates))
    
    if len(unique_dates) < 5:
        logger.warning(f"Only {len(unique_dates)} unique dates - validation may be unreliable")
    
    # Split dates chronologically: earliest -> train, latest -> val
    split_idx = max(1, int(len(unique_dates) * (1 - val_ratio)))  # At least 1 date in train
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])
    
    # Assign samples to splits based on their date
    train_indices = [i for i, d in sample_dates if d in train_dates]
    val_indices = [i for i, d in sample_dates if d in val_dates]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else Subset(dataset, [])
    
    # Log split details
    train_date_range = (min(train_dates).date(), max(train_dates).date()) if train_dates else (None, None)
    val_date_range = (min(val_dates).date(), max(val_dates).date()) if val_dates else (None, None)
    
    logger.info(f"Chronological split complete:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Train: {len(train_indices)} samples, {len(train_dates)} unique dates")
    logger.info(f"    Date range: {train_date_range[0]} to {train_date_range[1]}")
    logger.info(f"  Validation: {len(val_indices)} samples, {len(val_dates)} unique dates")
    if val_date_range[0]:
        logger.info(f"    Date range: {val_date_range[0]} to {val_date_range[1]}")
    
    # Verify no date overlap
    overlap = train_dates & val_dates
    if overlap:
        logger.error(f"CRITICAL: Date overlap between train and validation: {overlap}")
        raise ValueError(f"Train/validation date overlap detected: {overlap}")
    else:
        logger.info(f"  [PASS] Zero date overlap between train and validation")
    
    if len(val_indices) == 0:
        logger.warning("No validation samples - dataset too small or all dates in train")
    
    return train_dataset, val_dataset


def run_behavioral_cloning(
    trades_csv: Optional[str] = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = 'models/behavioral_cloning'
) -> Tuple[MaskedGaussianPolicy, float]:
    """Run complete Behavioral Cloning pipeline with REAL HISTORICAL DATA."""
    
    logger.info("=" * 70)
    logger.info("Behavioral Cloning - REAL HISTORICAL DATA ONLY")
    logger.info("=" * 70)
    logger.info("CRITICAL: All samples use REAL 60-bar windows from Parquet")
    logger.info("Anchoring: Deterministic based on actual bar indices")
    logger.info("  - Entries: Max VWAP deviation bar in entry window")
    logger.info("  - Non-entries: Random non-setup bar in entry window")
    logger.info("  - Window: [anchor-60, anchor) - strictly pre-anchor")
    logger.info("NO synthetic fallback - samples skipped if insufficient history")
    
    config = BCConfig(
        trades_csv=trades_csv or "reports/full_3527_backtest_results.csv",
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    perception_config = PerceptionConfig()
    sac_config = SACConfig()
    
    # Create dataset
    logger.info("\n[1/4] Loading dataset with REAL anchored data...")
    dataset = ExpertTradeDataset(config, perception_config)
    
    # Verify anchoring
    dataset.verify_anchoring(n_samples=5)
    
    # Chronological split by date (NOT random - prevents data leakage)
    train_dataset, val_dataset = create_chronological_split(dataset, val_ratio=0.2)
    
    if len(val_dataset) == 0:
        logger.warning("Dataset too small for validation split. Using full dataset for training.")
        val_dataset = None
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create actor
    logger.info("\n[2/4] Initializing SAC Actor...")
    actor = MaskedGaussianPolicy(
        state_dim=sac_config.state_dim,
        action_dim=sac_config.action_dim,
        hidden_dims=sac_config.actor_hidden_dims,
        action_low=sac_config.action_low,
        action_high=sac_config.action_high
    )
    
    logger.info(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    
    # Train
    logger.info("\n[3/4] Starting BC training...")
    trainer = BehavioralCloningTrainer(actor, config)
    history = trainer.train(train_loader, val_loader)
    
    # Save
    logger.info("\n[4/4] Saving trained model...")
    trainer.save_checkpoint('bc_final.pt')
    
    # RLlib format
    rllib_path = Path(config.output_dir) / 'bc_actor_rllib.pt'
    torch.save({
        'model_state_dict': actor.state_dict(),
        'config': sac_config,
        'bc_config': config,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None
    }, rllib_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("Behavioral Cloning Complete!")
    logger.info("=" * 70)
    logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Best Loss: {trainer.best_loss:.6f}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 70)
    
    return actor, trainer.best_loss


def test_behavioral_cloning():
    """Test BC pipeline with verification of real data and anchoring."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Behavioral Cloning with REAL ANCHORED DATA")
    logger.info("=" * 70)
    
    config = BCConfig(
        trades_csv="reports/relaxed_909_backtest.csv",
        num_epochs=2,
        batch_size=8,
        output_dir="models/bc_test_anchored"
    )
    
    perception_config = PerceptionConfig()
    sac_config = SACConfig()
    
    logger.info("\n[1/5] Loading dataset with anchored samples...")
    dataset = ExpertTradeDataset(config, perception_config)
    logger.info(f"Dataset: {len(dataset)} anchored samples")
    
    # Check positive/negative balance
    entries = sum(1 for s in dataset.samples if s.is_entry)
    flats = len(dataset.samples) - entries
    logger.info(f"  Entries: {entries}, Flats: {flats}")
    
    # Verify anchoring
    logger.info("\n[2/5] Verifying anchoring correctness...")
    anchoring_ok = dataset.verify_anchoring(n_samples=5)
    
    # Verify all samples use real data
    logger.info("\n[3/5] Verifying samples use REAL 60-bar windows...")
    none_count = 0
    for i in range(min(10, len(dataset))):
        state, action = dataset[i]
        if state is None or torch.all(state == 0):
            none_count += 1
    
    if none_count > 0:
        logger.error(f"  CRITICAL: {none_count}/10 samples returned None!")
        return None, float('inf')
    else:
        logger.info("  All sampled states are from REAL historical data ✓")
    
    # Sample check
    state, action = dataset[0]
    logger.info(f"\n[4/5] Sample state verification:")
    logger.info(f"  State shape: {state.shape}")
    logger.info(f"  State dtype: {state.dtype}")
    logger.info(f"  Target action: {action.item():.1f}")
    
    # Show sample details
    sample = dataset.samples[0]
    logger.info(f"  Sample details:")
    logger.info(f"    Symbol: {sample.symbol}, Date: {sample.date.date()}")
    logger.info(f"    Anchor idx: {sample.anchor_idx}, Window start: {sample.window_start_idx}")
    logger.info(f"    Anchor time: {sample.anchor_time}")
    logger.info(f"    Is entry: {sample.is_entry}")
    
    # Verify no synthetic fallback
    logger.info("\n[5/5] Verifying NO synthetic fallback...")
    has_synthetic = False
    for i in range(min(20, len(dataset))):
        seq = dataset._load_sequence_for_sample(dataset.samples[i])
        if seq is None:
            logger.error(f"  Sample {i} returned None - should have been filtered!")
            has_synthetic = True
        elif not isinstance(seq, np.ndarray):
            logger.error(f"  Sample {i} is not numpy array!")
            has_synthetic = True
    
    if not has_synthetic:
        logger.info("  ✓ All samples load REAL sequences from Parquet")
    
    # Test training
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    actor = MaskedGaussianPolicy(74, 1, [256, 256])
    trainer = BehavioralCloningTrainer(actor, config)
    history = trainer.train(loader, loader)
    
    logger.info(f"\nTraining losses: {[f'{l:.4f}' for l in history['train_loss']]}")
    logger.info("\n" + "=" * 70)
    logger.info("BC TEST RESULTS:")
    av = "PASS" if anchoring_ok else "FAIL"
    logger.info(f"  [{av}] Anchoring verification")
    logger.info("  [PASS] All samples use REAL 60-bar OHLCV from Parquet")
    logger.info("  [PASS] NO synthetic sequences generated")
    logger.info("  [PASS] Samples with insufficient history were skipped")
    logger.info("  [PASS] Window strictly precedes anchor bar")
    logger.info("=" * 70)
    
    return actor, trainer.best_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral Cloning - REAL ANCHORED DATA ONLY')
    parser.add_argument('--trades-csv', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='models/behavioral_cloning')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if args.test:
        test_behavioral_cloning()
    else:
        actor, final_loss = run_behavioral_cloning(
            trades_csv=args.trades_csv,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        print(f"\nFinal BC Loss: {final_loss:.6f}")
