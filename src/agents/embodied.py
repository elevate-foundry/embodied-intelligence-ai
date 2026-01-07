"""
Full Embodied Agent with World Model and Policy.

Combines:
- Transformer world model for predicting consequences
- RL policy for action selection
- External memory with degradation
- Consequence-driven learning
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import EmbodiedAgent
from .policy import ModelBasedPolicy, PolicyConfig
from ..models.transformer import WorldModelTrainer, TransformerConfig
from ..memory.external_memory import ExternalMemory


class TransformerAgent(EmbodiedAgent):
    """
    An embodied agent with:
    - Transformer world model for next-state prediction
    - PPO policy trained on top of world model
    - External memory with corruption
    
    This is the full implementation of the embodied intelligence architecture.
    """
    
    def __init__(
        self,
        observation_size: int,
        action_size: int = 6,
        memory_capacity: int = 32,
        memory_slot_size: int = 64,
        use_world_model: bool = True,
        use_planning: bool = True,
        planning_horizon: int = 3,
        seed: Optional[int] = None,
        device: str = "auto"
    ):
        super().__init__(
            observation_size=observation_size,
            action_size=action_size,
            memory_capacity=memory_capacity,
            memory_slot_size=memory_slot_size,
            seed=seed
        )
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerAgent")
        
        self.use_world_model = use_world_model
        self.use_planning = use_planning
        self.planning_horizon = planning_horizon
        
        # World model for predicting consequences
        if use_world_model:
            world_config = TransformerConfig(
                observation_size=observation_size,
                action_size=action_size
            )
            self.world_model = WorldModelTrainer(
                config=world_config,
                device=device
            )
        else:
            self.world_model = None
        
        # Policy for action selection
        policy_config = PolicyConfig(
            observation_size=observation_size,
            action_size=action_size
        )
        self.policy = ModelBasedPolicy(
            policy_config=policy_config,
            world_model_trainer=self.world_model,
            memory=self.memory,
            device=device
        )
        
        # Training state
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_value: float = 0.0
        self.last_log_prob: float = 0.0
        
        # Update frequency
        self.world_model_update_freq = 4
        self.policy_update_freq = 128
        self.steps_since_policy_update = 0
    
    def act(self, observation: np.ndarray) -> int:
        """Select an action using the policy."""
        # Get action from policy
        if self.use_planning and self.world_model is not None:
            action = self.policy.act_with_planning(
                observation,
                planning_horizon=self.planning_horizon
            )
        else:
            action = self.policy.act(observation)
        
        # Store for learning
        self.last_observation = observation.copy()
        self.last_action = action
        
        # Get value and log prob for PPO
        with torch.no_grad():
            obs = torch.from_numpy(observation).float().unsqueeze(0)
            obs = obs.to(self.policy.device) / 255.0
            _, log_prob, value = self.policy.policy.get_action(obs)
            self.last_value = value.item()
            self.last_log_prob = log_prob.item()
        
        return action
    
    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        """Learn from a transition."""
        self.total_reward += reward
        
        # Update world model
        if self.world_model is not None:
            self.world_model.add_transition(observation, action, next_observation)
            
            if self.total_steps % self.world_model_update_freq == 0:
                self.world_model.train_step()
        
        # Store transition for policy update
        self.policy.store_transition(
            observation=observation,
            action=action,
            reward=reward,
            value=self.last_value,
            log_prob=self.last_log_prob,
            done=done
        )
        
        self.steps_since_policy_update += 1
        
        # Update policy periodically
        if self.steps_since_policy_update >= self.policy_update_freq or done:
            # Get bootstrap value
            if done:
                last_value = 0.0
            else:
                with torch.no_grad():
                    obs = torch.from_numpy(next_observation).float().unsqueeze(0)
                    obs = obs.to(self.policy.device) / 255.0
                    _, _, value = self.policy.policy.get_action(obs)
                    last_value = value.item()
            
            self.policy.update(last_value)
            self.steps_since_policy_update = 0
        
        # Optionally store in external memory
        if reward > 1.0 or reward < -1.0:
            # Store significant transitions in memory
            self._store_to_memory(observation, action, reward)
    
    def _store_to_memory(self, observation: np.ndarray, action: int, reward: float):
        """Store significant experience in external memory."""
        # Find an empty or weak slot
        active = self.memory.get_active_slots()
        
        if len(active) < self.memory.capacity:
            # Use first empty slot
            for i in range(self.memory.capacity):
                if i not in active:
                    address = i
                    break
        else:
            # Overwrite weakest memory
            weakest = self.memory.get_strongest_memories(self.memory.capacity)
            address = weakest[-1][0]  # Weakest is last
        
        # Encode transition
        content = np.zeros(self.memory.slot_size, dtype=np.uint8)
        content[:len(observation)] = observation[:self.memory.slot_size - 2]
        content[-2] = action
        content[-1] = int(np.clip((reward + 10) * 10, 0, 255))  # Encode reward
        
        self.memory.write(address, content)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive statistics."""
        stats = super().get_stats()
        
        stats['policy'] = self.policy.get_stats()
        
        if self.world_model is not None:
            stats['world_model'] = self.world_model.get_stats()
        
        return stats
    
    def save(self, path: str):
        """Save agent checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.policy.save(os.path.join(path, 'policy.pt'))
        
        if self.world_model is not None:
            self.world_model.save(os.path.join(path, 'world_model.pt'))
    
    def load(self, path: str):
        """Load agent checkpoint."""
        import os
        
        self.policy.load(os.path.join(path, 'policy.pt'))
        
        if self.world_model is not None:
            self.world_model.load(os.path.join(path, 'world_model.pt'))
