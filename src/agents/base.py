"""
Base Embodied Agent.

An agent that:
- Perceives only through the substrate
- Maintains external memory (no magical recurrence)
- Acts through a fixed action vocabulary
- Learns from consequences, not explanations
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

from ..memory.external_memory import ExternalMemory


class EmbodiedAgent(ABC):
    """
    Abstract base class for embodied agents.
    
    All agents must:
    1. Receive observations only through the braille substrate
    2. Use external memory (no hidden recurrent state)
    3. Output actions from a fixed vocabulary
    4. Handle irreversibility (no state reloading)
    """
    
    def __init__(
        self,
        observation_size: int,
        action_size: int = 6,
        memory_capacity: int = 32,
        memory_slot_size: int = 64,
        seed: Optional[int] = None
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)
        
        # External memory - the only persistent state
        self.memory = ExternalMemory(
            capacity=memory_capacity,
            slot_size=memory_slot_size,
            seed=seed
        )
        
        # Statistics
        self.total_steps = 0
        self.total_reward = 0.0
        self.episode_count = 0
        
    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action given an observation.
        
        The observation is already encoded through the braille substrate.
        Must return an action index in [0, action_size).
        """
        pass
    
    @abstractmethod
    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        """
        Learn from a transition.
        
        This is where consequence-driven learning happens.
        The agent must update its policy based on outcomes.
        """
        pass
    
    def step(self):
        """Called after each environment step to update memory."""
        self.memory.step()
        self.total_steps += 1
    
    def reset_episode(self):
        """Called at the start of each episode."""
        self.episode_count += 1
        # Note: Memory persists across episodes (no free resets)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'episode_count': self.episode_count,
            'memory': self.memory.stats()
        }


class RandomAgent(EmbodiedAgent):
    """A baseline random agent for testing."""
    
    def act(self, observation: np.ndarray) -> int:
        return self.rng.integers(0, self.action_size)
    
    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        self.total_reward += reward
        # Random agent doesn't learn


class ReactiveAgent(EmbodiedAgent):
    """
    A simple reactive agent that responds to immediate observations.
    
    No planning, no world model - just stimulus-response.
    Useful as a baseline to compare against learned agents.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simple lookup table for reactions
        self.reaction_table: Dict[int, int] = {}
    
    def act(self, observation: np.ndarray) -> int:
        # Hash observation to get a key
        obs_key = hash(observation.tobytes()) % 10000
        
        if obs_key in self.reaction_table:
            return self.reaction_table[obs_key]
        else:
            # Default: random action
            return self.rng.integers(0, self.action_size)
    
    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        self.total_reward += reward
        
        # Simple learning: remember actions that led to positive reward
        if reward > 0:
            obs_key = hash(observation.tobytes()) % 10000
            self.reaction_table[obs_key] = action
        elif reward < -0.5:
            # Avoid actions that led to negative reward
            obs_key = hash(observation.tobytes()) % 10000
            if obs_key in self.reaction_table and self.reaction_table[obs_key] == action:
                del self.reaction_table[obs_key]
