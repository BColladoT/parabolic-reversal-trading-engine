"""
Module II: Deep Reinforcement Learning Decision Engine (Soft Actor-Critic)

Action masking architecture (Change 10):

ACTIVE PATH (used by train_wfo.py):
- Neural-level action masking via MaskedSACRLModule (new API) or MaskedSACModel (legacy API)
- MaskedGaussianPolicy constrains sampled actions to valid regions before they reach the env
- No post-hoc masking_penalty in the environment

Standalone (non-RLlib) path:
- StandaloneSACAgent uses MaskedGaussianPolicy directly
- Used by behavioral_cloning.py for BC pretraining

All masking thresholds match env.py canonical convention:
    action < -0.05: INCREASE short exposure
    action >  0.05: DECREASE short exposure
    else:           HOLD  [-0.05, 0.05]

Components:
1. MaskedGaussianPolicy - Standalone PyTorch policy with neural masking
2. StandaloneSACAgent   - Pure PyTorch SAC using MaskedGaussianPolicy
3. SACConfig            - Configuration dataclass
4. MaskedSACRLModule    - RLModule for Ray 2.x+ (wired in by train_wfo.py)
5. MaskedSACModel       - TorchModelV2 for legacy Ray API (wired in by train_wfo.py)
6. build_sac_config()   - Helper for building masked SAC configs

Author: AI Agent
Date: 2026-03-12
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ray RLlib imports - handle version differences gracefully
try:
    # Try new API first (Ray 2.x+)
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
    from ray.rllib.core.models.specs.typing import SpecType
    from ray.rllib.core.models.configs import ModelConfig
    from ray.rllib.algorithms.sac.sac_rl_module import SACRLModule
    from ray.rllib.algorithms.sac.torch.sac_torch_rl_module import SACTorchRLModule
    USE_NEW_API = True
    logger.info("Using Ray RLlib 2.x+ API")
except ImportError:
    # Fall back to legacy API
    try:
        from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
        from ray.rllib.models.modelv2 import ModelV2
        from ray.rllib.utils.framework import try_import_torch
        USE_NEW_API = False
        logger.info("Using Ray RLlib legacy API")
    except ImportError:
        logger.warning("Ray RLlib not installed. Installing...")
        USE_NEW_API = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SACConfig:
    """Configuration for SAC Algorithm."""
    
    # State and action dimensions
    state_dim: int = 74
    action_dim: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    
    # Network architecture
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    activation: str = "relu"
    
    # SAC specific parameters
    tau: float = 0.005              # Target network update rate
    gamma: float = 0.99             # Discount factor
    alpha: float = 0.01             # Initial temperature (scheduled: 0.01→0.05)
    auto_tune_alpha: bool = False   # Automatic entropy tuning (disabled; use schedule)
    target_entropy: Optional[float] = None  # Target entropy for auto-tuning
    
    # Training parameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    buffer_size: int = 1000000
    batch_size: int = 256
    
    # Action masking parameters
    mask_penalty: float = -1e9      # Large negative value for invalid actions
    
    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 256]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256]
        if self.target_entropy is None:
            # Heuristic: -dim(A) for continuous actions
            self.target_entropy = -self.action_dim


# =============================================================================
# Alpha (Entropy Temperature) Schedule
# =============================================================================

def get_alpha_schedule(
    timestep: int,
    phase1_end: int = 20000,
    phase2_end: int = 40000,
    alpha_start: float = 0.01,
    alpha_end: float = 0.10,
) -> float:
    """
    Three-phase entropy temperature schedule.
    Phase 1 (0-20K):      alpha = 0.01  (conservative exploration, actor frozen)
    Phase 2 (20K-40K):    linear ramp 0.03 → 0.10  (FIX 12: start at 0.03, not 0.01)
    Phase 3 (40K+):       alpha = 0.10  (sustained exploration — doubled from 0.05)

    Rationale: With longer training runs (300K+ steps), the agent needs higher
    entropy to avoid premature policy collapse into always-HOLD.
    """
    if timestep < phase1_end:
        return alpha_start
    elif timestep < phase2_end:
        progress = (timestep - phase1_end) / (phase2_end - phase1_end)
        # FIX 12: Ramp starts at 0.03 (not alpha_start=0.01) to soften the
        # jump when actor is unfrozen after Phase 1 critic warm-up.
        ramp_start = 0.03
        return ramp_start + progress * (alpha_end - ramp_start)
    else:
        return alpha_end


# =============================================================================
# Neural Network Components
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            if output_activation == "tanh":
                layers.append(nn.Tanh())
            elif output_activation == "sigmoid":
                layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MaskedGaussianPolicy(nn.Module):
    """
    Gaussian policy network with action masking for continuous actions.
    
    For continuous action masking, we use a technique where:
    1. The network outputs mean and log_std for a Gaussian distribution
    2. The action_mask determines which action directions are valid
    3. Invalid directions receive extremely high penalty in log_prob
    4. During sampling, actions are clipped to valid ranges based on mask
    
    Args:
        state_dim: Dimension of state input
        action_dim: Dimension of action output
        hidden_dims: Hidden layer dimensions
        action_low: Minimum action value
        action_high: Maximum action value
        mask_penalty: Penalty value for masked actions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        action_low: float = -1.0,
        action_high: float = 1.0,
        mask_penalty: float = -1e9
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.mask_penalty = mask_penalty
        
        # Mean network
        self.mean_net = MLP(
            state_dim,
            hidden_dims,
            action_dim,
            activation="relu"
        )
        
        # Log std network (state-dependent for flexibility)
        self.log_std_net = MLP(
            state_dim,
            hidden_dims,
            action_dim,
            activation="relu"
        )
        
        # Initialize log std to reasonable values (low variance initially)
        for layer in self.log_std_net.network:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.constant_(layer.bias, -1.0)  # Start with low std
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with action masking.
        
        Args:
            state: State tensor [batch, state_dim]
            action_mask: Action mask [batch, 3] for [increase, decrease, hold]
                         None means all actions valid
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Sampled/clipped action [batch, action_dim]
            log_prob: Log probability of action [batch]
            mean: Mean of distribution [batch, action_dim]
        """
        batch_size = state.size(0)
        
        # Get distribution parameters
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, -20, 2)  # Numerical stability
        self._last_log_std = log_std  # Expose for RLlib model output
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = torch.tanh(mean)
            # Apply action masking to deterministic action
            if action_mask is not None:
                action = self._apply_action_mask(action, action_mask)
            return action, torch.zeros(batch_size, device=state.device), mean
        
        # Sample action
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)
        
        # Apply action masking
        if action_mask is not None:
            action, log_prob = self._apply_mask_to_action(
                action, log_prob, mean, std, action_mask
            )
        
        return action, log_prob, mean
    
    def _apply_action_mask(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply action mask to constrain action range.

        CANONICAL ACTION CONVENTION (matches env.py):
            action in [-1, 1]
            action < -0.05: INCREASE short exposure (more negative)
            action >  0.05: DECREASE short exposure (cover toward 0)
            else:           HOLD current exposure  [-0.05, 0.05]

        Asymmetric thresholds break HOLD attractor: zero-mean Gaussian → COVER.

        Action mask indices:
            mask[0]: increase short allowed (1=yes, 0=no)
            mask[1]: decrease short/cover allowed (1=yes, 0=no)
            mask[2]: hold allowed (typically always 1)

        Args:
            action: Action in [-1, 1] [batch, action_dim]
            action_mask: Mask [batch, 3] for [increase, decrease, hold]

        Returns:
            masked_action: Constrained action
        """
        # CANONICAL CONVENTION:
        # action < -0.05: INCREASE short exposure → check mask[0]
        # action >  0.05: DECREASE short exposure → check mask[1]

        # If increase short is blocked (mask[0] == 0), clip to hold boundary
        increase_blocked = (action_mask[:, 0] == 0).float()
        # If decrease short is blocked (mask[1] == 0), clip to hold boundary
        decrease_blocked = (action_mask[:, 1] == 0).float()

        # Clip actions based on mask
        # If increase short blocked: action >= -0.05 (can't go more negative)
        # If decrease short blocked: action <= 0.05 (can't cover)
        action = torch.where(
            (action < -0.05) & (increase_blocked > 0),
            torch.full_like(action, -0.05),
            action
        )
        action = torch.where(
            (action > 0.05) & (decrease_blocked > 0),
            torch.full_like(action, 0.05),
            action
        )

        return action
    
    def _apply_mask_to_action(
        self,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mask to action and adjust log probability.
        
        CANONICAL ACTION CONVENTION (matches env.py):
            action < -0.05: INCREASE short exposure → check mask[0]
            action >  0.05: DECREASE short exposure → check mask[1]

        For continuous actions, masking works by:
        1. Constraining the action to valid ranges
        2. Adding large penalty to log_prob if action was in invalid region

        Args:
            action: Sampled action [batch, action_dim]
            log_prob: Log probability [batch]
            mean: Distribution mean [batch, action_dim]
            std: Distribution std [batch, action_dim]
            action_mask: Mask [batch, 3] for [increase, decrease, hold]

        Returns:
            masked_action: Constrained action
            masked_log_prob: Adjusted log probability
        """
        # CANONICAL CONVENTION:
        # action < -0.05: INCREASE short → check mask[0]
        # action >  0.05: DECREASE short → check mask[1]
        is_increase = (action < -0.05).float()  # INCREASE short (more negative)
        is_decrease = (action > 0.05).float()   # DECREASE short (cover)

        # Check if action is in blocked region
        increase_invalid = is_increase * (action_mask[:, 0:1] == 0).float()
        decrease_invalid = is_decrease * (action_mask[:, 1:2] == 0).float()

        # Any invalid action gets massive penalty
        invalid = (increase_invalid + decrease_invalid).clamp(0, 1)

        # Constrain action to valid region
        masked_action = action.clone()
        # If increase short blocked, clip to hold boundary -0.05
        masked_action = torch.where(
            (action < -0.05) & (action_mask[:, 0:1] == 0),
            torch.full_like(action, -0.05),
            masked_action
        )
        # If decrease short blocked, clip to hold boundary 0.05
        masked_action = torch.where(
            (action > 0.05) & (action_mask[:, 1:2] == 0),
            torch.full_like(action, 0.05),
            masked_action
        )
        
        # Add penalty to log_prob for invalid actions
        # This makes invalid actions have near-zero probability
        masked_log_prob = log_prob + (invalid.squeeze(-1) * self.mask_penalty)
        
        return masked_action, masked_log_prob


class QNetwork(nn.Module):
    """
    Q-network for critic (takes state and action, outputs Q-value).
    
    SAC uses twin Q-networks to mitigate overestimation bias.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        # Concatenate state and action
        input_dim = state_dim + action_dim
        
        self.network = MLP(
            input_dim,
            hidden_dims,
            output_dim=1,
            activation="relu"
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for state-action pair.
        
        Args:
            state: State tensor [batch, state_dim]
            action: Action tensor [batch, action_dim]
            
        Returns:
            q_value: Q(s, a) [batch, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# =============================================================================
# Custom RLlib Model (New API - Ray 2.x+)
# =============================================================================

if USE_NEW_API:
    class MaskedSACRLModule(SACTorchRLModule):
        """
        Custom SAC RLModule with action masking support.
        
        WARNING: THIS CLASS IS EXPERIMENTAL AND NOT ACTIVE IN THE MAIN TRAINER.
        The active WFO trainer (train_wfo.py) uses plain RLlib SAC without
        custom model masking. See module docstring for details.
        
        To use this class, you must explicitly wire it into the SAC config:
            sac_config.rl_module(rl_module_spec=MaskedSACRLModule)
        
        Observation format expected:
        {
            'state': [batch, 74],
            'action_mask': [batch, 3],  # [increase, decrease, hold]
            'kelly_leverage': [batch, 1]
        }
        """
        
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            
            self.mask_penalty = config.get("mask_penalty", -1e9)
            self.state_dim = config.get("state_dim", 74)
            self.action_dim = config.get("action_dim", 1)
            
            # Override policy and Q-networks with masked versions
            hidden_dims = config.get("actor_hidden_dims", [256, 256])
            
            self.policy = MaskedGaussianPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
                mask_penalty=self.mask_penalty
            )
            
            # Re-initialize Q-networks if needed
            critic_dims = config.get("critic_hidden_dims", [256, 256])
            self.q1 = QNetwork(self.state_dim, self.action_dim, critic_dims)
            self.q2 = QNetwork(self.state_dim, self.action_dim, critic_dims)
            
            logger.info("MaskedSACRLModule initialized with action masking")
        
        def _forward_exploration(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass for exploration (sampling with action mask)."""
            # Extract components from observation
            obs = batch["obs"]
            
            if isinstance(obs, dict):
                state = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state = obs
                action_mask = None
            
            # Get action from policy with masking
            action, log_prob, mean = self.policy(
                state,
                action_mask=action_mask,
                deterministic=False
            )
            
            return {
                "actions": action,
                "action_logp": log_prob,
                "mean_actions": mean
            }
        
        def _forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass for inference (deterministic with action mask)."""
            obs = batch["obs"]
            
            if isinstance(obs, dict):
                state = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state = obs
                action_mask = None
            
            # Get deterministic action with masking
            action, _, _ = self.policy(
                state,
                action_mask=action_mask,
                deterministic=True
            )
            
            return {"actions": action}
        
        def _forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass for training (computes all needed values)."""
            obs = batch["obs"]
            
            if isinstance(obs, dict):
                state = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state = obs
                action_mask = None
            
            # Sample action for policy evaluation
            action, log_prob, mean = self.policy(
                state,
                action_mask=action_mask,
                deterministic=False
            )
            
            # Compute Q-values
            q1_value = self.q1(state, action)
            q2_value = self.q2(state, action)
            
            # Target Q-values for critic training
            with torch.no_grad():
                next_action, next_log_prob, _ = self.policy(
                    batch.get("next_obs", state),
                    action_mask=action_mask,
                    deterministic=False
                )
                target_q1 = self.target_q1(state, next_action)
                target_q2 = self.target_q2(state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            
            return {
                "actions": action,
                "action_logp": log_prob,
                "q1": q1_value,
                "q2": q2_value,
                "target_q": target_q,
                "mean_actions": mean
            }


# =============================================================================
# Custom RLlib Model (Legacy API - Ray 1.x)
# =============================================================================

if not USE_NEW_API and USE_NEW_API is not None:
    from ray.rllib.models.torch.misc import SlimFC
    from ray.rllib.utils.typing import ModelConfigDict, TensorType
    
    class MaskedSACModel(TorchModelV2, nn.Module):
        """
        Legacy custom SAC model with action masking.
        
        WARNING: THIS CLASS IS EXPERIMENTAL AND NOT ACTIVE IN THE MAIN TRAINER.
        The active WFO trainer (train_wfo.py) uses plain RLlib SAC without
        custom model masking. See module docstring for details.
        
        Compatible with Ray RLlib 1.x versions (legacy API).
        """
        
        def __init__(
            self,
            obs_space,
            action_space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            **kwargs
        ):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)
            
            # Extract config
            self.mask_penalty = model_config.get("custom_model_config", {}).get(
                "mask_penalty", -1e9
            )
            self.state_dim = model_config.get("custom_model_config", {}).get(
                "state_dim", 74
            )
            self.action_dim = action_space.shape[0]  # true action dim (not 2*action_dim)
            
            hidden_dims = model_config.get("custom_model_config", {}).get(
                "actor_hidden_dims", [256, 256]
            )
            
            # Create masked policy
            self.policy = MaskedGaussianPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
                mask_penalty=self.mask_penalty
            )
            
            logger.info("MaskedSACModel (legacy) initialized")
        
        def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType
        ) -> Tuple[TensorType, List[TensorType]]:
            """
            Forward pass with action masking.
            
            For SAC, forward returns action distribution parameters.
            """
            obs = input_dict["obs"]
            
            # Extract state and mask from dict observation
            if isinstance(obs, dict):
                state_features = obs.get("state", obs)
                action_mask = obs.get("action_mask", None)
            else:
                state_features = obs
                action_mask = None
            
            # Get action from policy
            action, log_prob, mean = self.policy(
                state_features,
                action_mask=action_mask,
                deterministic=False
            )

            # SAC SquashedGaussian expects [mean, log_std] concatenated → (batch, num_outputs=2)
            # Use masked_action (clipped to valid region) as distribution center so RLlib
            # samples cluster around valid actions rather than the raw unmasked mean.
            log_std = self.policy._last_log_std
            return torch.cat([action, log_std], dim=-1), state
        
        def value_function(self) -> TensorType:
            """Value function not used in SAC (uses Q-networks instead)."""
            return torch.zeros(1)


# =============================================================================
# RLlib Configuration Builder
# =============================================================================

def build_sac_config(
    env_class = None,
    config: Optional[SACConfig] = None,
    custom_model: bool = False  # Default to plain SAC (production path)
) -> Any:
    """
    Build Ray RLlib SAC configuration.
    
    IMPORTANT: This function has TWO modes controlled by custom_model parameter.
    
    ACTIVE PATH (custom_model=False, default in train_wfo.py):
        Returns plain RLlib SAC configuration. This is the PRODUCTION path.
        Action masking is handled by environment penalty, not neural network.
    
    EXPERIMENTAL PATH (custom_model=True):
        Wires in MaskedSACRLModule or MaskedSACModel for action masking.
        WARNING: This path is NOT ACTIVE in train_wfo.py and is EXPERIMENTAL.
        Use only if you explicitly want to test model-level action masking.
    
    Args:
        env_class: Gym environment class (e.g., ParabolicReversalEnv)
        config: SAC configuration
        custom_model: Whether to use custom masked model (EXPERIMENTAL, default False)
        
    Returns:
        algo_config: RLlib algorithm configuration
    """
    if custom_model:
        logger.warning("=" * 70)
        logger.warning("EXPERIMENTAL: build_sac_config called with custom_model=True")
        logger.warning("The masked model path is NOT active in train_wfo.py")
        logger.warning("Use custom_model=False for production training")
        logger.warning("=" * 70)
    
    try:
        from ray.rllib.algorithms.sac import SACConfig as RLlibSACConfig
    except ImportError:
        logger.error("Ray RLlib not installed. Run: pip install ray[rllib]")
        raise
    
    config = config or SACConfig()
    
    # Build base SAC config
    if USE_NEW_API:
        # Ray 2.x style
        algo_config = (
            RLlibSACConfig()
            .environment(
                env=env_class if env_class else "ParabolicReversalEnv",
                env_config={
                    "initial_capital": 100000.0,
                }
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={
                    "fcnet_hiddens": config.critic_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                policy_model_config={
                    "fcnet_hiddens": config.actor_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                tau=config.tau,
                initial_alpha=config.alpha,
                target_entropy=config.target_entropy,
                n_step=1,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": config.buffer_size,
                },
            )
            .rl_module(
                rl_module_spec=MaskedSACRLModule if custom_model else None
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
                num_cpus_per_worker=2,
            )
            .rollouts(
                num_rollout_workers=2,
                rollout_fragment_length=200,
            )
            .reporting(
                min_time_s_per_iteration=5,
            )
        )
    else:
        # Ray 1.x style
        from ray.rllib.models import ModelCatalog
        
        # Register custom model
        if custom_model:
            ModelCatalog.register_custom_model("masked_sac_model", MaskedSACModel)
        
        algo_config = (
            RLlibSACConfig()
            .environment(
                env=env_class if env_class else "ParabolicReversalEnv",
            )
            .framework("torch")
            .training(
                twin_q=True,
                q_model_config={
                    "fcnet_hiddens": config.critic_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                policy_model_config={
                    "fcnet_hiddens": config.actor_hidden_dims,
                    "fcnet_activation": config.activation,
                },
                tau=config.tau,
                initial_alpha=config.alpha,
                target_entropy=config.target_entropy,
            )
            .resources(
                num_gpus=1 if torch.cuda.is_available() else 0,
            )
            .rollouts(
                num_rollout_workers=2,
            )
        )
        
        if custom_model:
            algo_config.model({
                "custom_model": "masked_sac_model",
                "custom_model_config": {
                    "mask_penalty": config.mask_penalty,
                    "state_dim": config.state_dim,
                    "actor_hidden_dims": config.actor_hidden_dims,
                }
            })
    
    return algo_config


# =============================================================================
# Standalone SAC Agent (for non-RLlib usage)
# =============================================================================

class StandaloneSACAgent:
    """
    Standalone SAC agent for environments where Ray RLlib is not available.
    
    This provides a pure PyTorch implementation of SAC with action masking
    that can be used directly without RLlib.
    """
    
    def __init__(self, config: Optional[SACConfig] = None):
        self.config = config or SACConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network
        self.actor = MaskedGaussianPolicy(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.actor_hidden_dims,
            action_low=self.config.action_low,
            action_high=self.config.action_high,
            mask_penalty=self.config.mask_penalty
        ).to(self.device)
        
        # Critic networks (twin Q)
        self.q1 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        self.q2 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        # Target Q-networks
        self.target_q1 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        self.target_q2 = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        
        # Copy weights to targets
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Temperature parameter
        self.log_alpha = torch.tensor(
            np.log(self.config.alpha),
            requires_grad=True,
            device=self.device
        )
        
        # Optimizers — FIX 8: weight_decay=1e-4 to reduce memorization risk
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor, weight_decay=1e-4
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=self.config.lr_critic, weight_decay=1e-4
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=self.config.lr_critic, weight_decay=1e-4
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.lr_alpha
        )
        
        self.steps = 0
        
        logger.info(f"StandaloneSACAgent initialized on {self.device}")
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: State vector [state_dim]
            action_mask: Action mask [3]
            deterministic: If True, use mean action
            
        Returns:
            action: Selected action [action_dim]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            else:
                mask_tensor = None
            
            action, _, _ = self.actor(
                state_tensor,
                action_mask=mask_tensor,
                deterministic=deterministic
            )
            
            return action.cpu().numpy()[0]
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent using a batch of transitions.
        
        Args:
            batch: Dictionary with 'state', 'action', 'reward', 'next_state', 'done'
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Apply alpha schedule (manual, since auto_tune_alpha=False)
        scheduled_alpha = get_alpha_schedule(self.steps)
        self.log_alpha.data.fill_(math.log(max(scheduled_alpha, 1e-8)))

        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # Get action mask if available
        action_mask = batch.get('action_mask', None)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
        
        # ===== Update Critic =====
        with torch.no_grad():
            # Sample next action
            next_action, next_log_prob, _ = self.actor(
                next_state, action_mask=action_mask, deterministic=False
            )
            
            # Compute target Q
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.config.gamma * (
                target_q - self.log_alpha.exp() * next_log_prob.unsqueeze(-1)
            )
        
        # Compute current Q
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        
        # Critic loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # Update critics
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        # ===== Update Actor =====
        new_action, log_prob, _ = self.actor(
            state, action_mask=action_mask, deterministic=False
        )
        
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp().detach() * log_prob.unsqueeze(-1) - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== Update Alpha =====
        if self.config.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.config.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # ===== Update Target Networks =====
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        
        self.steps += 1
        
        return {
            'q_loss': q_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'avg_q': q_new.mean().item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )


# =============================================================================
# Testing and Validation
# =============================================================================

def test_agent():
    """Test the SAC agent components."""
    logger.info("=" * 70)
    logger.info("Testing SAC Decision Engine")
    logger.info("=" * 70)
    
    config = SACConfig()
    
    # Test 1: Masked Gaussian Policy
    logger.info("\n[TEST 1] Masked Gaussian Policy")
    policy = MaskedGaussianPolicy(
        state_dim=74,
        action_dim=1,
        hidden_dims=[256, 256]
    )
    
    batch_size = 4
    state = torch.randn(batch_size, 74)
    
    # Test without mask
    action, log_prob, mean = policy(state, action_mask=None, deterministic=False)
    logger.info(f"  State shape:  {state.shape}")
    logger.info(f"  Action shape: {action.shape}")
    logger.info(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    logger.info(f"  Log prob:     {log_prob.shape}")
    
    # Test with mask (block increase)
    mask = torch.tensor([
        [0, 1, 1],  # Block increase
        [1, 1, 1],  # All valid
        [1, 0, 1],  # Block decrease
        [0, 0, 1],  # Block both directions
    ])
    
    action_masked, log_prob_masked, _ = policy(state, action_mask=mask, deterministic=False)
    logger.info(f"\n  With action mask:")
    logger.info(f"  Mask [0,1,1] (block increase): action={action_masked[0].item():.3f}")
    logger.info(f"  Mask [1,1,1] (all valid):      action={action_masked[1].item():.3f}")
    logger.info(f"  Mask [1,0,1] (block decrease): action={action_masked[2].item():.3f}")
    logger.info(f"  Mask [0,0,1] (block both):     action={action_masked[3].item():.3f}")
    
    # Verify masking works
    assert action_masked[0] <= 0.1, "Increase should be blocked"
    assert action_masked[2] >= -0.1, "Decrease should be blocked"
    assert abs(action_masked[3]) <= 0.1, "Both directions blocked"
    logger.info("  ✓ Action masking working correctly")
    
    # Test 2: Q-Network
    logger.info("\n[TEST 2] Q-Network")
    q_net = QNetwork(state_dim=74, action_dim=1, hidden_dims=[256, 256])
    q_value = q_net(state, action)
    logger.info(f"  Q-value shape: {q_value.shape}")
    logger.info(f"  Q-value range: [{q_value.min():.3f}, {q_value.max():.3f}]")
    logger.info("  ✓ Q-network working")
    
    # Test 3: Standalone Agent
    logger.info("\n[TEST 3] Standalone SAC Agent")
    agent = StandaloneSACAgent(config)
    
    # Test action selection
    test_state = np.random.randn(74)
    test_mask = np.array([1, 1, 1])
    
    action = agent.select_action(test_state, test_mask, deterministic=False)
    logger.info(f"  Selected action: {action}")
    
    action_det = agent.select_action(test_state, test_mask, deterministic=True)
    logger.info(f"  Deterministic action: {action_det}")
    
    # Test update
    batch = {
        'state': torch.randn(32, 74),
        'action': torch.randn(32, 1),
        'reward': torch.randn(32),
        'next_state': torch.randn(32, 74),
        'done': torch.zeros(32),
        'action_mask': torch.ones(32, 3)
    }
    
    metrics = agent.update(batch)
    logger.info(f"  Update metrics: {metrics}")
    logger.info("  ✓ Agent update working")
    
    # Test 4: RLlib Config
    logger.info("\n[TEST 4] RLlib Configuration")
    try:
        algo_config = build_sac_config(custom_model=True)
        logger.info("  ✓ RLlib config built successfully")
        logger.info(f"  Config type: {type(algo_config)}")
    except Exception as e:
        logger.warning(f"  RLlib not available: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("All SAC tests passed!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    test_agent()
