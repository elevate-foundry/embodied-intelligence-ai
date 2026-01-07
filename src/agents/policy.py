"""
RL Policy trained on top of World Model.

The policy uses the world model to:
1. Simulate future trajectories
2. Evaluate action consequences
3. Select actions that maximize survival and goal completion
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..models.transformer import TransformerConfig, WorldModelTrainer
from ..memory.external_memory import ExternalMemory


@dataclass
class PolicyConfig:
    """Configuration for the RL policy."""
    observation_size: int = 25
    action_size: int = 6
    hidden_dim: int = 128
    num_layers: int = 2
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clip
    entropy_coef: float = 0.01    # Entropy bonus
    value_coef: float = 0.5       # Value loss coefficient
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4


if TORCH_AVAILABLE:
    
    class PolicyNetwork(nn.Module):
        """
        Actor-Critic policy network.
        
        Takes braille-encoded observations and outputs:
        - Action probabilities (actor)
        - State value estimate (critic)
        """
        
        def __init__(self, config: PolicyConfig):
            super().__init__()
            self.config = config
            
            # Observation encoder
            self.obs_encoder = nn.Sequential(
                nn.Linear(config.observation_size, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU()
            )
            
            # Memory encoder (optional - for using external memory)
            self.memory_encoder = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU()
            )
            
            # Actor head
            self.actor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.action_size)
            )
            
            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1)
            )
        
        def forward(
            self,
            observation: torch.Tensor,
            memory_state: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                observation: (batch, obs_size) normalized observation
                memory_state: Optional (batch, memory_dim) memory encoding
                
            Returns:
                action_logits: (batch, action_size)
                value: (batch, 1)
            """
            # Encode observation
            hidden = self.obs_encoder(observation)
            
            # Optionally incorporate memory
            if memory_state is not None:
                mem_encoded = self.memory_encoder(memory_state)
                hidden = hidden + F.pad(mem_encoded, (0, hidden.size(-1) - mem_encoded.size(-1)))
            
            # Actor and critic outputs
            action_logits = self.actor(hidden)
            value = self.critic(hidden)
            
            return action_logits, value
        
        def get_action(
            self,
            observation: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample an action from the policy.
            
            Returns:
                action: Selected action
                log_prob: Log probability of action
                value: State value estimate
            """
            action_logits, value = self.forward(observation)
            
            dist = Categorical(logits=action_logits)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action, log_prob, value.squeeze(-1)
        
        def evaluate_actions(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for PPO update.
            
            Returns:
                log_probs: Log probabilities of actions
                values: State value estimates
                entropy: Policy entropy
            """
            action_logits, values = self.forward(observations)
            
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            return log_probs, values.squeeze(-1), entropy


class ModelBasedPolicy:
    """
    RL policy that uses the world model for planning.
    
    Combines:
    1. Direct policy learning (actor-critic)
    2. Model-based rollouts for action evaluation
    3. External memory for long-term context
    """
    
    def __init__(
        self,
        policy_config: Optional[PolicyConfig] = None,
        world_model_trainer: Optional[WorldModelTrainer] = None,
        memory: Optional[ExternalMemory] = None,
        device: str = "auto"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelBasedPolicy")
        
        self.config = policy_config or PolicyConfig()
        self.world_model = world_model_trainer
        self.memory = memory
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create policy network
        self.policy = PolicyNetwork(self.config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )
        
        # Experience buffer for PPO
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        
        # Stats
        self.update_count = 0
        self.total_reward = 0.0
    
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> int:
        """Select an action given an observation."""
        self.policy.eval()
        
        with torch.no_grad():
            # Normalize observation
            obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            obs = obs / 255.0  # Normalize 8-bit tokens
            
            action, log_prob, value = self.policy.get_action(obs, deterministic)
            
        return action.item()
    
    def act_with_planning(
        self,
        observation: np.ndarray,
        planning_horizon: int = 3,
        num_simulations: int = 10
    ) -> int:
        """
        Select action using model-based planning.
        
        Simulates future trajectories using the world model
        and selects the action with best expected outcome.
        """
        if self.world_model is None:
            return self.act(observation)
        
        best_action = 0
        best_value = float('-inf')
        
        for action in range(self.config.action_size):
            total_value = 0.0
            
            for _ in range(num_simulations):
                value = self._simulate_trajectory(
                    observation, action, planning_horizon
                )
                total_value += value
            
            avg_value = total_value / num_simulations
            
            if avg_value > best_value:
                best_value = avg_value
                best_action = action
        
        return best_action
    
    def _simulate_trajectory(
        self,
        start_obs: np.ndarray,
        first_action: int,
        horizon: int
    ) -> float:
        """Simulate a trajectory using the world model."""
        obs = start_obs.copy()
        total_reward = 0.0
        discount = 1.0
        
        # Take first action
        next_obs = self.world_model.predict(obs, first_action)
        
        # Simple reward heuristic based on observation changes
        # In practice, you'd want a learned reward model
        reward = self._estimate_reward(obs, first_action, next_obs)
        total_reward += discount * reward
        
        obs = next_obs
        
        # Continue with policy actions
        for _ in range(horizon - 1):
            discount *= self.config.gamma
            
            action = self.act(obs, deterministic=True)
            next_obs = self.world_model.predict(obs, action)
            
            reward = self._estimate_reward(obs, action, next_obs)
            total_reward += discount * reward
            
            obs = next_obs
        
        return total_reward
    
    def _estimate_reward(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray
    ) -> float:
        """
        Estimate reward from transition.
        
        This is a simple heuristic. A full implementation would
        learn a reward model or use the environment's reward.
        """
        # Reward for change (exploration)
        change = np.sum(obs != next_obs) / len(obs)
        
        # Penalty for staying still (encourages movement)
        if action == 4:  # wait
            return -0.1
        
        return change * 0.1
    
    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a transition for PPO update."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.total_reward += reward
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update on collected experience.
        
        Returns training statistics.
        """
        if len(self.observations) == 0:
            return {}
        
        # Compute advantages using GAE
        advantages = self._compute_gae(last_value)
        returns = advantages + np.array(self.values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.from_numpy(np.array(self.observations)).float().to(self.device)
        obs_tensor = obs_tensor / 255.0
        act_tensor = torch.from_numpy(np.array(self.actions)).long().to(self.device)
        old_log_probs = torch.from_numpy(np.array(self.log_probs)).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        
        # PPO update
        self.policy.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(4):  # PPO epochs
            log_probs, values, entropy = self.policy.evaluate_actions(obs_tensor, act_tensor)
            
            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns_tensor)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss 
                + self.config.value_coef * value_loss 
                + self.config.entropy_coef * entropy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear buffer
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        self.update_count += 1
        
        return {
            'policy_loss': total_policy_loss / 4,
            'value_loss': total_value_loss / 4,
            'entropy': total_entropy / 4,
            'update_count': self.update_count
        }
    
    def _compute_gae(self, last_value: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
        
        return advantages
    
    def save(self, path: str):
        """Save policy checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load policy checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
    
    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics."""
        return {
            'update_count': self.update_count,
            'total_reward': self.total_reward,
            'buffer_size': len(self.observations),
            'device': str(self.device)
        }
