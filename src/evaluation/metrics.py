"""
Evaluation Metrics for Embodied Intelligence.

Success is not benchmark scores.
Success is observing behaviors that:
- Anticipate consequences
- Recover from self-caused damage
- Adapt strategies when the world changes
- Fail in ways that make sense
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""
    episode_id: int
    survived: bool
    goal_reached: bool
    steps: int
    total_reward: float
    energy_remaining: int
    hazards_hit: int
    energy_collected: int
    termination_reason: str
    trajectory: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for embodied agents.
    
    Measures:
    1. Survival - Can the agent stay alive?
    2. Task completion - Can it reach goals?
    3. Efficiency - How quickly and with what resources?
    4. Adaptation - Does it improve over time?
    5. Generalization - Does it work in new environments?
    """
    
    # Core metrics
    survival_rate: float = 0.0
    goal_completion_rate: float = 0.0
    average_reward: float = 0.0
    average_steps: float = 0.0
    
    # Efficiency metrics
    average_energy_remaining: float = 0.0
    energy_efficiency: float = 0.0  # Reward per energy spent
    step_efficiency: float = 0.0    # Reward per step
    
    # Consequence handling
    hazard_avoidance_rate: float = 0.0
    recovery_rate: float = 0.0  # Survived after hitting hazard
    
    # Adaptation metrics
    early_survival_rate: float = 0.0   # First 25% of episodes
    late_survival_rate: float = 0.0    # Last 25% of episodes
    improvement_rate: float = 0.0      # Late - Early
    
    # Generalization
    train_performance: float = 0.0
    test_performance: float = 0.0
    generalization_gap: float = 0.0
    
    # Episode details
    num_episodes: int = 0
    episode_results: List[EpisodeResult] = field(default_factory=list)
    
    def compute_from_results(self, results: List[EpisodeResult]):
        """Compute all metrics from episode results."""
        if not results:
            return
        
        self.num_episodes = len(results)
        self.episode_results = results
        
        # Core metrics
        self.survival_rate = sum(1 for r in results if r.survived) / len(results)
        self.goal_completion_rate = sum(1 for r in results if r.goal_reached) / len(results)
        self.average_reward = np.mean([r.total_reward for r in results])
        self.average_steps = np.mean([r.steps for r in results])
        
        # Efficiency
        self.average_energy_remaining = np.mean([r.energy_remaining for r in results])
        
        total_energy_spent = sum(100 - r.energy_remaining for r in results)
        total_reward = sum(r.total_reward for r in results)
        self.energy_efficiency = total_reward / max(1, total_energy_spent)
        
        total_steps = sum(r.steps for r in results)
        self.step_efficiency = total_reward / max(1, total_steps)
        
        # Hazard handling
        episodes_with_hazards = [r for r in results if r.hazards_hit > 0]
        if episodes_with_hazards:
            self.recovery_rate = sum(1 for r in episodes_with_hazards if r.survived) / len(episodes_with_hazards)
        
        episodes_without_hazards = [r for r in results if r.hazards_hit == 0]
        self.hazard_avoidance_rate = len(episodes_without_hazards) / len(results)
        
        # Adaptation (learning curve)
        quarter = max(1, len(results) // 4)
        early_results = results[:quarter]
        late_results = results[-quarter:]
        
        self.early_survival_rate = sum(1 for r in early_results if r.survived) / len(early_results)
        self.late_survival_rate = sum(1 for r in late_results if r.survived) / len(late_results)
        self.improvement_rate = self.late_survival_rate - self.early_survival_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'survival_rate': self.survival_rate,
            'goal_completion_rate': self.goal_completion_rate,
            'average_reward': self.average_reward,
            'average_steps': self.average_steps,
            'average_energy_remaining': self.average_energy_remaining,
            'energy_efficiency': self.energy_efficiency,
            'step_efficiency': self.step_efficiency,
            'hazard_avoidance_rate': self.hazard_avoidance_rate,
            'recovery_rate': self.recovery_rate,
            'early_survival_rate': self.early_survival_rate,
            'late_survival_rate': self.late_survival_rate,
            'improvement_rate': self.improvement_rate,
            'train_performance': self.train_performance,
            'test_performance': self.test_performance,
            'generalization_gap': self.generalization_gap,
            'num_episodes': self.num_episodes
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "EMBODIED INTELLIGENCE EVALUATION",
            "=" * 50,
            "",
            "Core Performance:",
            f"  Survival Rate:        {self.survival_rate:.1%}",
            f"  Goal Completion:      {self.goal_completion_rate:.1%}",
            f"  Average Reward:       {self.average_reward:.2f}",
            f"  Average Steps:        {self.average_steps:.1f}",
            "",
            "Efficiency:",
            f"  Energy Remaining:     {self.average_energy_remaining:.1f}",
            f"  Energy Efficiency:    {self.energy_efficiency:.4f}",
            f"  Step Efficiency:      {self.step_efficiency:.4f}",
            "",
            "Consequence Handling:",
            f"  Hazard Avoidance:     {self.hazard_avoidance_rate:.1%}",
            f"  Recovery Rate:        {self.recovery_rate:.1%}",
            "",
            "Adaptation:",
            f"  Early Survival:       {self.early_survival_rate:.1%}",
            f"  Late Survival:        {self.late_survival_rate:.1%}",
            f"  Improvement:          {self.improvement_rate:+.1%}",
            "",
        ]
        
        if self.generalization_gap != 0:
            lines.extend([
                "Generalization:",
                f"  Train Performance:    {self.train_performance:.2f}",
                f"  Test Performance:     {self.test_performance:.2f}",
                f"  Generalization Gap:   {self.generalization_gap:.2f}",
                ""
            ])
        
        lines.append(f"Total Episodes: {self.num_episodes}")
        lines.append("=" * 50)
        
        return "\n".join(lines)


def evaluate_agent(
    agent,
    environment,
    num_episodes: int = 100,
    max_steps: int = 500,
    seed: Optional[int] = None,
    verbose: bool = False
) -> EvaluationMetrics:
    """
    Evaluate an agent on an environment.
    
    Args:
        agent: The agent to evaluate
        environment: The environment to evaluate in
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        EvaluationMetrics with comprehensive results
    """
    results = []
    
    for ep in range(num_episodes):
        # Reset
        if seed is not None:
            obs = environment.reset(seed=seed + ep)
        else:
            obs = environment.reset()
        
        agent.reset_episode()
        
        result = EpisodeResult(
            episode_id=ep,
            survived=True,
            goal_reached=False,
            steps=0,
            total_reward=0.0,
            energy_remaining=100,
            hazards_hit=0,
            energy_collected=0,
            termination_reason=""
        )
        
        done = False
        
        while not done and result.steps < max_steps:
            # Agent acts
            action = agent.act(obs)
            
            # Environment responds
            next_obs, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            
            # Track trajectory
            result.trajectory.append((environment.agent.x, environment.agent.y))
            
            # Update result
            result.total_reward += reward
            result.steps += 1
            
            if info.get('goal_reached'):
                result.goal_reached = True
            if info.get('hazard'):
                result.hazards_hit += 1
            if info.get('energy_collected'):
                result.energy_collected += 1
            
            # Learn (optional during evaluation)
            agent.learn(obs, action, reward, next_obs, done)
            agent.step()
            
            obs = next_obs
        
        # Final state
        result.survived = environment.agent.alive
        result.energy_remaining = environment.agent.energy
        result.termination_reason = info.get('reason', 'max_steps')
        
        results.append(result)
        
        if verbose and (ep + 1) % 10 == 0:
            recent = results[-10:]
            survival = sum(1 for r in recent if r.survived) / len(recent)
            goals = sum(1 for r in recent if r.goal_reached) / len(recent)
            print(f"Episode {ep + 1}/{num_episodes} | Survival: {survival:.1%} | Goals: {goals:.1%}")
    
    # Compute metrics
    metrics = EvaluationMetrics()
    metrics.compute_from_results(results)
    
    return metrics


def evaluate_generalization(
    agent,
    train_env,
    test_envs: List,
    num_episodes_per_env: int = 50,
    verbose: bool = False
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate agent generalization across different environments.
    
    Args:
        agent: The agent to evaluate
        train_env: The training environment
        test_envs: List of test environments with different properties
        num_episodes_per_env: Episodes per environment
        verbose: Print progress
        
    Returns:
        Dictionary mapping environment name to metrics
    """
    results = {}
    
    # Evaluate on training environment
    if verbose:
        print("Evaluating on training environment...")
    train_metrics = evaluate_agent(
        agent, train_env, 
        num_episodes=num_episodes_per_env,
        verbose=verbose
    )
    results['train'] = train_metrics
    
    # Evaluate on test environments
    for i, test_env in enumerate(test_envs):
        if verbose:
            print(f"Evaluating on test environment {i + 1}...")
        test_metrics = evaluate_agent(
            agent, test_env,
            num_episodes=num_episodes_per_env,
            verbose=verbose
        )
        results[f'test_{i + 1}'] = test_metrics
    
    # Compute generalization gap
    train_perf = train_metrics.average_reward
    test_perfs = [m.average_reward for k, m in results.items() if k.startswith('test')]
    avg_test_perf = np.mean(test_perfs) if test_perfs else 0.0
    
    for metrics in results.values():
        metrics.train_performance = train_perf
        metrics.test_performance = avg_test_perf
        metrics.generalization_gap = train_perf - avg_test_perf
    
    return results
