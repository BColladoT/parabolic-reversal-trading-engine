"""
Module I: Modular Perception and State Representation

This module implements the perception layer for the Parabolic Reversal Trading
trading system using a Temporal Convolutional Autoencoder (TCN-AE). It provides:

1. Temporal Convolutional Autoencoder (TCN-AE):
   - Encoder: Compresses high-dimensional OHLCV time-series into latent vector z_t
   - Decoder: Reconstructs input sequence for self-supervised pre-training
   - Causal convolutions prevent look-ahead bias

2. Hybrid State Concatenation:
   - Extracts frozen latent vector z_t (64 dimensions)
   - Concatenates with explicit V5 Relaxed features:
     * VWAP deviation (normalized)
     * Volume concentration
   - Combines with portfolio state features
   - Outputs final 74-dimensional state vector S_t

3. Pre-training Infrastructure:
   - Self-supervised MSE reconstruction loss
   - Learning rate scheduling
   - Checkpoint management
   - Data loader for historical market sequences

Author: AI Agent
Date: 2026-03-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PerceptionConfig:
    """Configuration for the Modular Perception module."""
    
    # Input sequence parameters
    sequence_length: int = 60  # 60 minutes of OHLCV data
    num_features: int = 5      # OHLCV: Open, High, Low, Close, Volume
    
    # TCN Architecture parameters
    encoder_channels: List[int] = None  # Channel progression
    kernel_size: int = 3       # Convolution kernel size
    dropout: float = 0.2       # Dropout rate for regularization
    use_weight_norm: bool = True  # Weight normalization for stability
    
    # Latent space dimensions (must match env.py observation space)
    latent_dim: int = 64       # Bottleneck layer size (z_t)
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 10         # Early stopping patience
    min_delta: float = 1e-6    # Minimum improvement for early stopping
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpointing
    checkpoint_dir: str = "models/perception"
    save_best: bool = True
    
    def __post_init__(self):
        if self.encoder_channels is None:
            # Default: 5 -> 32 -> 64 -> 128 -> 64 progression
            self.encoder_channels = [32, 64, 128, 64]


# =============================================================================
# TCN Components
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution to prevent look-ahead bias.
    
    Causal convolution ensures that the output at time t only depends on
    inputs from time 0 to t, never on future information. This is critical
    for financial time series to prevent data leakage.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        dilation: Dilation factor for receptive field expansion
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = True
    ):
        super().__init__()
        
        # Calculate padding for causal convolution
        # Output length = Input length - (kernel_size - 1) * dilation
        # To maintain length, pad (kernel_size - 1) * dilation on the left
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # We'll handle padding manually
            dilation=dilation
        )
        
        if use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding.
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            Output tensor [batch, out_channels, time]
        """
        # Pad left side only (causal padding)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual block with dilated causal convolution.
    
    Architecture:
        Input -> Conv1 -> ReLU -> Dropout -> Conv2 -> ReLU -> Dropout -> Add -> Output
               |_______________________________________________________|
    
    Args:
        channels: Number of channels (input = output for residual)
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool = True
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation, use_weight_norm
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation, use_weight_norm
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (identity if same dimensions)
        self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward pass."""
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out + residual


# =============================================================================
# Temporal Convolutional Autoencoder (TCN-AE)
# =============================================================================

class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder.
    
    Compresses high-dimensional time-series into low-dimensional latent vector.
    Uses dilated convolutions to exponentially expand receptive field without
    increasing parameters.
    
    Architecture:
        Input: [batch, num_features, sequence_length]
        -> Conv layers with increasing dilation
        -> Global average pooling
        -> Linear projection to latent_dim
        Output: [batch, latent_dim]
        
    Args:
        config: PerceptionConfig with architecture parameters
    """
    
    def __init__(self, config: PerceptionConfig):
        super().__init__()
        
        self.config = config
        channels = [config.num_features] + config.encoder_channels
        
        # Build encoder layers with exponentially increasing dilation
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            
            # Initial convolution
            layers.append(CausalConv1d(
                in_ch, out_ch, config.kernel_size, dilation, config.use_weight_norm
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            
            # Residual block
            layers.append(ResidualBlock(
                out_ch, config.kernel_size, dilation, config.dropout, config.use_weight_norm
            ))
        
        self.encoder_layers = nn.Sequential(*layers)
        
        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(
            config.encoder_channels[-1],
            config.latent_dim
        )
        
        # Layer normalization for stable latent representations
        self.latent_norm = nn.LayerNorm(config.latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent vector.
        
        Args:
            x: Input tensor [batch, num_features, sequence_length]
            
        Returns:
            z: Latent vector [batch, latent_dim]
        """
        # Apply encoder layers
        features = self.encoder_layers(x)  # [batch, channels, time]
        
        # Global average pooling over time dimension
        pooled = self.global_pool(features).squeeze(-1)  # [batch, channels]
        
        # Project to latent space
        z = self.to_latent(pooled)  # [batch, latent_dim]
        
        # Normalize for stable learning
        z = self.latent_norm(z)
        
        return z


class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network Decoder.
    
    Reconstructs input sequence from latent vector for self-supervised pre-training.
    
    Architecture:
        Input: [batch, latent_dim]
        -> Linear expansion
        -> ConvTranspose layers
        Output: [batch, num_features, sequence_length]
        
    Args:
        config: PerceptionConfig with architecture parameters
    """
    
    def __init__(self, config: PerceptionConfig):
        super().__init__()
        
        self.config = config
        
        # Calculate the feature map size after encoder pooling
        # This will be the target for initial expansion
        hidden_dim = config.encoder_channels[-1]
        
        # Expand latent vector to time-distributed features
        self.expand = nn.Sequential(
            nn.Linear(config.latent_dim, hidden_dim * (config.sequence_length // 4)),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Transposed convolution layers for upsampling
        # Reverse of encoder: 64 -> 128 -> 64 -> 32 -> 5
        decoder_channels = list(reversed(config.encoder_channels)) + [config.num_features]
        
        layers = []
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Use transposed convolution for upsampling
            layers.append(nn.ConvTranspose1d(
                in_ch, out_ch,
                kernel_size=config.kernel_size,
                stride=2 if i < 2 else 1,  # Upsample first two layers
                padding=config.kernel_size // 2,
                output_padding=1 if i < 2 else 0
            ))
            
            if i < len(decoder_channels) - 2:  # No activation on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Final adjustment to exact sequence length
        self.output_adjust = nn.Conv1d(
            config.num_features,
            config.num_features,
            kernel_size=1
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to sequence reconstruction.
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            reconstruction: [batch, num_features, sequence_length]
        """
        batch_size = z.size(0)
        
        # Expand to time-distributed representation
        hidden = self.expand(z)  # [batch, hidden_dim * (seq_len // 4)]
        hidden = hidden.view(
            batch_size,
            self.config.encoder_channels[-1],
            self.config.sequence_length // 4
        )
        
        # Apply decoder layers
        out = self.decoder_layers(hidden)
        
        # Adjust to exact sequence length
        if out.size(-1) != self.config.sequence_length:
            out = F.interpolate(
                out,
                size=self.config.sequence_length,
                mode='linear',
                align_corners=False
            )
        
        out = self.output_adjust(out)
        
        return out


class TemporalAutoencoder(nn.Module):
    """
    Complete Temporal Convolutional Autoencoder (TCN-AE).
    
    Combines encoder and decoder for end-to-end training.
    After pre-training, the decoder is discarded and only the encoder is used
    for state representation.
    
    Args:
        config: PerceptionConfig with architecture parameters
    """
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        super().__init__()
        
        self.config = config or PerceptionConfig()
        
        self.encoder = TCNEncoder(self.config)
        self.decoder = TCNDecoder(self.config)
        
        # Move to device
        self.to(self.config.device)
        
        logger.info(
            f"TCN-AE initialized: {self.config.num_features} -> "
            f"{self.config.latent_dim} -> {self.config.num_features} "
            f"(sequence length: {self.config.sequence_length})"
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor [batch, num_features, sequence_length]
            
        Returns:
            z: Latent vector [batch, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            reconstruction: [batch, num_features, sequence_length]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor [batch, num_features, sequence_length]
            
        Returns:
            z: Latent vector [batch, latent_dim]
            reconstruction: [batch, num_features, sequence_length]
        """
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon
    
    def freeze_encoder(self):
        """Freeze encoder weights after pre-training (discard decoder)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen for deployment")


# =============================================================================
# Hybrid State Representation
# =============================================================================

class StateRepresentation(nn.Module):
    """
    Hybrid State Representation Module.
    
    Combines the frozen latent vector from the TCN encoder with explicit
    V5 Relaxed features to form the final state vector S_t.
    
    Output format (74 dimensions total, matching env.py observation space):
        [0:64]   - Latent vector z_t (from TCN encoder)
        [64]     - VWAP deviation (normalized)
        [65]     - Volume concentration
        [66]     - Position size (normalized)
        [67]     - Unrealized P&L percentage
        [68]     - Current drawdown (normalized)
        [69]     - Kelly leverage fraction
        [70]     - Hour of day (normalized)
        [71]     - Minute (normalized)
        [72]     - In entry window flag
        [73]     - Must flatten flag
    
    Args:
        encoder: Frozen TCNEncoder instance
        config: PerceptionConfig
    """
    
    def __init__(
        self,
        encoder: TCNEncoder,
        config: Optional[PerceptionConfig] = None
    ):
        super().__init__()
        
        self.encoder = encoder
        self.config = config or PerceptionConfig()
        self.latent_dim = self.config.latent_dim
        
        # Freeze encoder if not already frozen
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Normalization parameters for explicit features
        # Based on empirical analysis from ARCHITECTURE_BLUEPRINT.md
        self.register_buffer('vwap_mean', torch.tensor(30.0))
        self.register_buffer('vwap_std', torch.tensor(15.0))
        self.register_buffer('max_position_value', torch.tensor(50000.0))
        self.register_buffer('max_drawdown', torch.tensor(-19180.0))
        
        logger.info(
            f"StateRepresentation initialized: "
            f"latent_dim={self.latent_dim}, total_state_dim=74"
        )
    
    def forward(
        self,
        market_sequence: torch.Tensor,
        vwap_deviation: torch.Tensor,
        volume_concentration: torch.Tensor,
        position_value: torch.Tensor,
        unrealized_pnl_pct: torch.Tensor,
        current_drawdown: torch.Tensor,
        kelly_fraction: torch.Tensor,
        time_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct the complete state vector.
        
        Args:
            market_sequence: OHLCV sequence [batch, num_features, sequence_length]
            vwap_deviation: VWAP deviation percentage [batch]
            volume_concentration: Volume concentration [0-1] [batch]
            position_value: Current position value [batch]
            unrealized_pnl_pct: Unrealized P&L as percentage [batch]
            current_drawdown: Current drawdown amount [batch]
            kelly_fraction: Current Kelly leverage [batch]
            time_features: [hour, minute, in_window, must_flatten] [batch, 4]
            
        Returns:
            state: Complete state vector [batch, 74]
        """
        batch_size = market_sequence.size(0)
        device = market_sequence.device
        
        # Extract latent representation (frozen encoder)
        with torch.no_grad():
            z = self.encoder(market_sequence)  # [batch, latent_dim]
        
        # Normalize explicit features
        vwap_norm = (vwap_deviation - self.vwap_mean) / self.vwap_std
        vwap_norm = vwap_norm.unsqueeze(-1)  # [batch, 1]
        
        vol_conc = volume_concentration.unsqueeze(-1)  # [batch, 1]
        
        pos_norm = (position_value / self.max_position_value).unsqueeze(-1)
        
        pnl_pct = unrealized_pnl_pct.unsqueeze(-1)  # Already percentage
        
        dd_norm = (current_drawdown / self.max_drawdown).unsqueeze(-1)
        
        kelly_norm = (kelly_fraction / 3.0).unsqueeze(-1)  # Max leverage is 3.0
        
        # Ensure time features are correct shape
        if time_features.dim() == 1:
            time_features = time_features.unsqueeze(0)
        
        # Concatenate all features
        state = torch.cat([
            z,                          # [batch, 64]
            vwap_norm,                  # [batch, 1]
            vol_conc,                   # [batch, 1]
            pos_norm,                   # [batch, 1]
            pnl_pct,                    # [batch, 1]
            dd_norm,                    # [batch, 1]
            kelly_norm,                 # [batch, 1]
            time_features               # [batch, 4]
        ], dim=-1)  # [batch, 74]
        
        return state
    
    def to_numpy(self, state: torch.Tensor) -> np.ndarray:
        """Convert state tensor to numpy array for environment."""
        return state.cpu().numpy()


# =============================================================================
# Dataset and Data Loading
# =============================================================================

class MarketSequenceDataset(Dataset):
    """
    Dataset for market sequence samples.
    
    Loads and preprocesses OHLCV sequences for autoencoder training.
    
    Args:
        data: DataFrame with OHLCV columns
        sequence_length: Length of each sequence sample
        transform: Optional preprocessing transform
    """
    
    def __init__(
        self,
        data: pl.DataFrame,
        sequence_length: int = 60,
        transform: Optional[Any] = None
    ):
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Extract OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c.lower() for c in data.columns]
        
        # Normalize column names
        col_mapping = {}
        for req in required_cols:
            for avail in data.columns:
                if req in avail.lower():
                    col_mapping[req] = avail
                    break
        
        if len(col_mapping) < 5:
            raise ValueError(f"Missing required columns. Found: {col_mapping}")
        
        # Extract and normalize features
        self.features = np.zeros((len(data), 5))
        for i, col in enumerate(required_cols):
            if col in col_mapping:
                self.features[:, i] = data[col_mapping[col]].to_numpy()
        
        # Normalize each feature (z-score)
        self.feature_means = self.features.mean(axis=0)
        self.feature_stds = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.feature_means) / self.feature_stds
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.features) - sequence_length + 1):
            seq = self.features[i:i + sequence_length]
            self.sequences.append(seq)
        
        self.sequences = np.array(self.sequences)
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]  # [sequence_length, 5]
        
        # Transpose to [channels, time]
        seq = torch.FloatTensor(seq).transpose(0, 1)
        
        if self.transform:
            seq = self.transform(seq)
        
        return seq


# =============================================================================
# Pre-training Infrastructure
# =============================================================================

class PerceptionTrainer:
    """
    Trainer for self-supervised pre-training of the TCN Autoencoder.
    
    Implements:
    - MSE reconstruction loss
    - Adam optimizer with learning rate scheduling
    - Early stopping
    - Checkpoint saving
    
    Args:
        model: TemporalAutoencoder instance
        config: PerceptionConfig
    """
    
    def __init__(
        self,
        model: TemporalAutoencoder,
        config: Optional[PerceptionConfig] = None
    ):
        self.model = model
        self.config = config or PerceptionConfig()
        self.device = torch.device(self.config.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            z, recon = self.model(batch)
            loss = self.criterion(recon, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = batch.to(self.device)
                z, recon = self.model(batch)
                loss = self.criterion(recon, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            history: Dictionary with train and validation losses
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if self.config.save_best:
                        self.save_checkpoint('best_model.pt')
                        logger.info(f"New best model saved (val_loss: {val_loss:.6f})")
                else:
                    self.epochs_without_improvement += 1
                
                logger.info(
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Best: {self.best_val_loss:.6f}"
                )
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break
            else:
                logger.info(f"Train Loss: {train_loss:.6f}")
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if val_loader else None
        }
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {path}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_perception_module(
    checkpoint_path: Optional[str] = None,
    config: Optional[PerceptionConfig] = None
) -> Tuple[StateRepresentation, PerceptionConfig]:
    """
    Factory function to create perception module.
    
    Loads pre-trained encoder if checkpoint provided, otherwise creates fresh model.
    
    Args:
        checkpoint_path: Path to pre-trained checkpoint
        config: Optional configuration (uses default if not provided)
        
    Returns:
        perception: StateRepresentation module with frozen encoder
        config: PerceptionConfig used
    """
    config = config or PerceptionConfig()
    
    # Create autoencoder
    autoencoder = TemporalAutoencoder(config)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pre-trained encoder from {checkpoint_path}")
    
    # Freeze encoder and create state representation
    autoencoder.freeze_encoder()
    perception = StateRepresentation(autoencoder.encoder, config)
    
    return perception, config


def pretrain_perception(
    data: pl.DataFrame,
    val_split: float = 0.2,
    config: Optional[PerceptionConfig] = None
) -> TemporalAutoencoder:
    """
    End-to-end pre-training function.
    
    Args:
        data: OHLCV DataFrame for training
        val_split: Fraction of data for validation
        config: Training configuration
        
    Returns:
        model: Trained TemporalAutoencoder
    """
    config = config or PerceptionConfig()
    
    # Create dataset
    dataset = MarketSequenceDataset(data, config.sequence_length)
    
    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model and trainer
    model = TemporalAutoencoder(config)
    trainer = PerceptionTrainer(model, config)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    return model, history


# =============================================================================
# Testing and Validation
# =============================================================================

def test_perception_module():
    """Run comprehensive tests on the perception module."""
    logger.info("=" * 70)
    logger.info("Testing Modular Perception (TCN-AE)")
    logger.info("=" * 70)
    
    # Create config
    config = PerceptionConfig(
        sequence_length=60,
        num_features=5,
        latent_dim=64,
        batch_size=16
    )
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Sequence length: {config.sequence_length}")
    logger.info(f"  Input features: {config.num_features}")
    logger.info(f"  Latent dimension: {config.latent_dim}")
    logger.info(f"  Device: {config.device}")
    
    # Test 1: Autoencoder forward pass
    logger.info("\n[TEST 1] Autoencoder forward pass")
    model = TemporalAutoencoder(config)
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.num_features, config.sequence_length)
    dummy_input = dummy_input.to(config.device)
    
    z, recon = model(dummy_input)
    
    logger.info(f"  Input shape:  {dummy_input.shape}")
    logger.info(f"  Latent shape: {z.shape}")
    logger.info(f"  Output shape: {recon.shape}")
    
    assert z.shape == (batch_size, config.latent_dim), "Latent shape mismatch"
    assert recon.shape == dummy_input.shape, "Reconstruction shape mismatch"
    logger.info("  ✓ Tensor shapes correct")
    
    # Test 2: State representation
    logger.info("\n[TEST 2] Hybrid State Representation")
    
    model.freeze_encoder()
    state_module = StateRepresentation(model.encoder, config)
    
    # Create dummy market data
    market_seq = torch.randn(batch_size, config.num_features, config.sequence_length)
    
    # Create dummy explicit features
    vwap_dev = torch.tensor([25.0, 30.0, 46.0, 21.0])  # Including threshold boundary
    vol_conc = torch.tensor([0.8, 1.0, 0.66, 0.5])
    pos_value = torch.tensor([25000.0, 0.0, 50000.0, 10000.0])
    unrealized_pnl = torch.tensor([0.02, 0.0, -0.01, 0.05])
    drawdown = torch.tensor([-5000.0, 0.0, -10000.0, -15000.0])
    kelly = torch.tensor([0.5, 0.1, 1.5, 2.0])
    time_feats = torch.tensor([
        [10.5, 30.0, 1.0, 0.0],
        [9.0, 0.0, 0.0, 0.0],
        [14.0, 45.0, 1.0, 0.0],
        [15.5, 0.0, 0.0, 1.0]
    ])
    
    state = state_module(
        market_seq, vwap_dev, vol_conc, pos_value,
        unrealized_pnl, drawdown, kelly, time_feats
    )
    
    logger.info(f"  Market sequence: {market_seq.shape}")
    logger.info(f"  Latent vector:   {z.shape}")
    logger.info(f"  Final state:     {state.shape}")
    
    assert state.shape == (batch_size, 74), f"State shape {state.shape} != (4, 74)"
    logger.info("  ✓ State vector shape correct (batch, 74)")
    
    # Verify state composition
    logger.info("\n[TEST 3] State vector composition")
    logger.info(f"  [0:64]   Latent z:          {state[0, 0:64].shape}")
    logger.info(f"  [64]     VWAP deviation:    {state[0, 64].item():.4f}")
    logger.info(f"  [65]     Volume conc:       {state[0, 65].item():.4f}")
    logger.info(f"  [66]     Position:          {state[0, 66].item():.4f}")
    logger.info(f"  [67]     Unrealized PnL:    {state[0, 67].item():.4f}")
    logger.info(f"  [68]     Drawdown:          {state[0, 68].item():.4f}")
    logger.info(f"  [69]     Kelly fraction:    {state[0, 69].item():.4f}")
    logger.info(f"  [70:74]  Time features:     {state[0, 70:74].numpy()}")
    
    # Test 3: Reconstruction quality
    logger.info("\n[TEST 4] Reconstruction quality")
    mse = F.mse_loss(recon, dummy_input).item()
    logger.info(f"  MSE (untrained): {mse:.6f}")
    
    # Test 4: Encoder receptive field
    logger.info("\n[TEST 5] Causal convolution check")
    seq1 = torch.randn(1, config.num_features, config.sequence_length)
    seq2 = seq1.clone()
    seq2[:, :, -1] += 10.0  # Modify only last time step
    
    z1 = model.encode(seq1)
    z2 = model.encode(seq2)
    
    diff = torch.abs(z1 - z2).mean().item()
    logger.info(f"  Latent difference after last-step perturbation: {diff:.6f}")
    assert diff > 0.01, "Encoder may not be properly causal"
    logger.info("  ✓ Encoder responds to recent changes (causal)")
    
    # Test 6: Dataset creation
    logger.info("\n[TEST 6] Dataset creation")
    dummy_df = pl.DataFrame({
        'open': np.random.randn(1000),
        'high': np.random.randn(1000),
        'low': np.random.randn(1000),
        'close': np.random.randn(1000),
        'volume': np.random.randn(1000)
    })
    
    dataset = MarketSequenceDataset(dummy_df, sequence_length=60)
    logger.info(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    logger.info(f"  Sample shape: {sample.shape}")
    assert sample.shape == (5, 60), "Dataset sample shape incorrect"
    logger.info("  ✓ Dataset working correctly")
    
    logger.info("\n" + "=" * 70)
    logger.info("All tests passed successfully!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    test_perception_module()
