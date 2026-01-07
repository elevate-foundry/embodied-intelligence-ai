"""
Training Loop for Embodied Intelligence.

The closed-loop interaction:
    observe → decide → act → world changes → observe

All intelligence emerges from surviving and succeeding within this loop.
"""

import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import time


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_id: int
    total_reward: float = 0.0
    steps: int = 0
    survived: bool = True
    goal_reached: bool = False
    energy_collected: int = 0
    hazards_hit: int = 0
    duration_seconds: float = 0.0
    termination_reason: str = ""


@dataclass
class TrainingStats:
    """Aggregate training statistics."""
    total_episodes: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    survival_rate: float = 0.0
    goal_completion_rate: float = 0.0
    average_episode_length: float = 0.0
    episode_history: List[EpisodeStats] = field(default_factory=list)


class TrainingLoop:
    """
    The core training loop for embodied intelligence.
    
    Implements the closed-loop interaction pattern:
    1. Agent observes world through substrate
    2. Agent decides on action
    3. Action is executed, world changes
    4. Consequences are observed
    5. Agent learns from consequences
    """
    
    def __init__(
        self,
        environment,
        agent,
        world_model=None,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 500,
        log_interval: int = 10,
        seed: Optional[int] = None
    ):
        self.env = environment
        self.agent = agent
        self.world_model = world_model
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.log_interval = log_interval
        
        self.rng = np.random.default_rng(seed)
        self.stats = TrainingStats()
        self._running = False
        
    def run(self, verbose: bool = True) -> TrainingStats:
        """
        Run the full training loop.
        
        Returns aggregate statistics.
        """
        self._running = True
        
        for episode in range(self.max_episodes):
            if not self._running:
                break
                
            episode_stats = self._run_episode(episode)
            self._update_stats(episode_stats)
            
            if verbose and (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1)
        
        return self.stats
    
    def _run_episode(self, episode_id: int) -> EpisodeStats:
        """Run a single episode."""
        start_time = time.time()
        
        # Reset environment (the only allowed reset)
        observation = self.env.reset()
        self.agent.reset_episode()
        
        stats = EpisodeStats(episode_id=episode_id)
        done = False
        
        while not done and stats.steps < self.max_steps_per_episode:
            # 1. Agent decides on action
            action = self.agent.act(observation)
            
            # 2. Execute action, world changes
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 3. Update world model with observed transition
            if self.world_model is not None:
                self.world_model.update(observation, action, next_observation)
            
            # 4. Agent learns from consequences
            self.agent.learn(observation, action, reward, next_observation, done)
            
            # 5. Update memory (degradation happens here)
            self.agent.step()
            
            # Track statistics
            stats.total_reward += reward
            stats.steps += 1
            
            if info.get('goal_reached'):
                stats.goal_reached = True
            if info.get('energy_collected'):
                stats.energy_collected += 1
            if info.get('hazard'):
                stats.hazards_hit += 1
            
            # Prepare for next iteration
            observation = next_observation
        
        # Episode ended
        stats.duration_seconds = time.time() - start_time
        stats.survived = self.env.agent.alive
        stats.termination_reason = info.get('reason', 'max_steps')
        
        return stats
    
    def _update_stats(self, episode_stats: EpisodeStats):
        """Update aggregate statistics."""
        self.stats.total_episodes += 1
        self.stats.total_steps += episode_stats.steps
        self.stats.total_reward += episode_stats.total_reward
        self.stats.episode_history.append(episode_stats)
        
        # Calculate rates
        survived = sum(1 for e in self.stats.episode_history if e.survived)
        goals = sum(1 for e in self.stats.episode_history if e.goal_reached)
        
        self.stats.survival_rate = survived / self.stats.total_episodes
        self.stats.goal_completion_rate = goals / self.stats.total_episodes
        self.stats.average_episode_length = self.stats.total_steps / self.stats.total_episodes
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        recent = self.stats.episode_history[-self.log_interval:]
        recent_reward = sum(e.total_reward for e in recent) / len(recent)
        recent_survival = sum(1 for e in recent if e.survived) / len(recent)
        recent_goals = sum(1 for e in recent if e.goal_reached) / len(recent)
        
        print(f"Episode {episode}/{self.max_episodes} | "
              f"Reward: {recent_reward:.2f} | "
              f"Survival: {recent_survival:.1%} | "
              f"Goals: {recent_goals:.1%} | "
              f"Steps: {self.stats.total_steps}")
    
    def stop(self):
        """Stop the training loop."""
        self._running = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        return {
            'total_episodes': self.stats.total_episodes,
            'total_steps': self.stats.total_steps,
            'total_reward': self.stats.total_reward,
            'survival_rate': self.stats.survival_rate,
            'goal_completion_rate': self.stats.goal_completion_rate,
            'average_episode_length': self.stats.average_episode_length,
            'agent_stats': self.agent.get_stats(),
            'world_model_stats': self.world_model.get_stats() if self.world_model else None
        }
