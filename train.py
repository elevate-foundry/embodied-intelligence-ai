#!/usr/bin/env python3
"""
Embodied Intelligence AI - Full Training Script

Trains:
1. Transformer world model as next-state predictor
2. RL policy on top of world model
3. Evaluates survival, task completion, and generalization
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional

from src.environments import BrailleGridWorld
from src.agents.base import RandomAgent, ReactiveAgent
from src.evaluation.metrics import evaluate_agent, EvaluationMetrics

# Check for PyTorch
try:
    import torch
    from src.agents.embodied import TransformerAgent
    from src.models.transformer import WorldModelTrainer, TransformerConfig
    from src.agents.policy import ModelBasedPolicy, PolicyConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TransformerAgent = None
    WorldModelTrainer = None
    TransformerConfig = None
    print("Warning: PyTorch not available. Using baseline agents only.")


def create_environment(
    width: int = 10,
    height: int = 10,
    view_radius: int = 2,
    difficulty: str = "medium",
    seed: int = None
) -> BrailleGridWorld:
    """Create environment with specified difficulty."""
    
    difficulty_settings = {
        "easy": {
            "energy_cost_per_step": 1,
            "hazard_damage": 10,
            "stochastic": False,
            "noise_level": 0.0
        },
        "medium": {
            "energy_cost_per_step": 1,
            "hazard_damage": 20,
            "stochastic": False,
            "noise_level": 0.02
        },
        "hard": {
            "energy_cost_per_step": 2,
            "hazard_damage": 30,
            "stochastic": True,
            "noise_level": 0.05
        }
    }
    
    settings = difficulty_settings.get(difficulty, difficulty_settings["medium"])
    
    return BrailleGridWorld(
        width=width,
        height=height,
        view_radius=view_radius,
        seed=seed,
        **settings
    )


def train_world_model_only(
    env: BrailleGridWorld,
    num_episodes: int = 500,
    verbose: bool = True
) -> WorldModelTrainer:
    """
    Phase 1: Train world model to predict next states.
    
    Uses random exploration to collect transitions.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for world model training")
    
    print("\n" + "=" * 60)
    print("PHASE 1: WORLD MODEL TRAINING")
    print("Learning to predict consequences of actions")
    print("=" * 60 + "\n")
    
    observation_size = (2 * env.view_radius + 1) ** 2
    
    config = TransformerConfig(
        observation_size=observation_size,
        action_size=6,
        hidden_dim=128,
        num_layers=4,
        num_heads=4
    )
    
    trainer = WorldModelTrainer(config=config)
    
    # Collect transitions with random policy
    random_agent = RandomAgent(observation_size=observation_size, seed=42)
    
    total_transitions = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            action = random_agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add to world model
            trainer.add_transition(obs, action, next_obs)
            total_transitions += 1
            
            # Train periodically
            if total_transitions % 4 == 0:
                trainer.train_step()
            
            obs = next_obs
        
        if verbose and (ep + 1) % 50 == 0:
            stats = trainer.get_stats()
            print(f"Episode {ep + 1}/{num_episodes} | "
                  f"Transitions: {total_transitions} | "
                  f"Avg Loss: {stats['average_loss']:.4f}")
    
    print(f"\nWorld model training complete!")
    print(f"Total transitions collected: {total_transitions}")
    print(f"Training steps: {trainer.train_steps}")
    
    return trainer


def train_full_agent(
    env: BrailleGridWorld,
    num_episodes: int = 1000,
    use_planning: bool = True,
    verbose: bool = True
) -> TransformerAgent:
    """
    Phase 2: Train full agent with world model and policy.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for full agent training")
    
    print("\n" + "=" * 60)
    print("PHASE 2: FULL AGENT TRAINING")
    print("Learning policy on top of world model")
    print("=" * 60 + "\n")
    
    observation_size = (2 * env.view_radius + 1) ** 2
    
    agent = TransformerAgent(
        observation_size=observation_size,
        action_size=6,
        memory_capacity=32,
        use_world_model=True,
        use_planning=use_planning,
        planning_horizon=3,
        seed=42
    )
    
    # Training loop
    episode_rewards = []
    episode_survivals = []
    episode_goals = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        agent.reset_episode()
        
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.learn(obs, action, reward, next_obs, done)
            agent.step()
            
            episode_reward += reward
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_survivals.append(env.agent.alive)
        episode_goals.append(info.get('goal_reached', False))
        
        if verbose and (ep + 1) % 100 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_survival = np.mean(episode_survivals[-100:])
            recent_goals = np.mean(episode_goals[-100:])
            
            stats = agent.get_stats()
            wm_stats = stats.get('world_model', {})
            
            print(f"Episode {ep + 1}/{num_episodes} | "
                  f"Reward: {recent_reward:.2f} | "
                  f"Survival: {recent_survival:.1%} | "
                  f"Goals: {recent_goals:.1%} | "
                  f"WM Loss: {wm_stats.get('average_loss', 0):.4f}")
    
    print(f"\nAgent training complete!")
    print(f"Final survival rate: {np.mean(episode_survivals[-100:]):.1%}")
    print(f"Final goal rate: {np.mean(episode_goals[-100:]):.1%}")
    
    return agent


def run_evaluation(
    agent,
    train_env: BrailleGridWorld,
    num_episodes: int = 100,
    test_generalization: bool = True,
    verbose: bool = True
) -> EvaluationMetrics:
    """
    Phase 3: Comprehensive evaluation.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: EVALUATION")
    print("Measuring survival, task completion, and generalization")
    print("=" * 60 + "\n")
    
    # Evaluate on training environment
    print("Evaluating on training environment...")
    train_metrics = evaluate_agent(
        agent, train_env,
        num_episodes=num_episodes,
        verbose=verbose
    )
    
    print("\n" + train_metrics.summary())
    
    if test_generalization:
        print("\nTesting generalization on harder environments...")
        
        # Create harder test environments
        test_envs = [
            create_environment(width=12, height=12, difficulty="hard", seed=999),
            create_environment(width=8, height=8, difficulty="hard", seed=888),
        ]
        
        for i, test_env in enumerate(test_envs):
            print(f"\nTest environment {i + 1}:")
            test_metrics = evaluate_agent(
                agent, test_env,
                num_episodes=num_episodes // 2,
                verbose=False
            )
            print(f"  Survival: {test_metrics.survival_rate:.1%}")
            print(f"  Goals: {test_metrics.goal_completion_rate:.1%}")
            print(f"  Avg Reward: {test_metrics.average_reward:.2f}")
    
    return train_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Embodied Intelligence AI")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--no-planning", action="store_true", help="Disable model-based planning")
    parser.add_argument("--save-path", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline agents")
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          EMBODIED INTELLIGENCE AI - TRAINING              ║")
    print("║                                                            ║")
    print("║  Intelligence that emerges from surviving in a world      ║")
    print("║  with consequences.                                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Set seeds
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
    
    # Create training environment
    train_env = create_environment(
        width=10,
        height=10,
        view_radius=2,
        difficulty=args.difficulty,
        seed=args.seed
    )
    
    observation_size = (2 * train_env.view_radius + 1) ** 2
    print(f"Environment: {train_env.width}x{train_env.height} grid")
    print(f"Observation size: {observation_size} tokens")
    print(f"Difficulty: {args.difficulty}")
    print()
    
    # Baseline evaluation
    print("=" * 60)
    print("BASELINE: Random Agent")
    print("=" * 60)
    
    random_agent = RandomAgent(observation_size=observation_size, seed=args.seed)
    random_metrics = evaluate_agent(
        random_agent, train_env,
        num_episodes=args.eval_episodes,
        verbose=True
    )
    print(f"\nRandom Agent - Survival: {random_metrics.survival_rate:.1%}, "
          f"Goals: {random_metrics.goal_completion_rate:.1%}")
    
    print("\n" + "=" * 60)
    print("BASELINE: Reactive Agent")
    print("=" * 60)
    
    reactive_agent = ReactiveAgent(observation_size=observation_size, seed=args.seed)
    # Train reactive agent briefly
    for _ in range(args.episodes // 2):
        obs = train_env.reset()
        done = False
        while not done:
            action = reactive_agent.act(obs)
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            reactive_agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
    
    reactive_metrics = evaluate_agent(
        reactive_agent, train_env,
        num_episodes=args.eval_episodes,
        verbose=True
    )
    print(f"\nReactive Agent - Survival: {reactive_metrics.survival_rate:.1%}, "
          f"Goals: {reactive_metrics.goal_completion_rate:.1%}")
    
    if args.baseline_only or not TORCH_AVAILABLE:
        print("\nBaseline evaluation complete.")
        return
    
    # Full training
    start_time = time.time()
    
    # Phase 1: World model
    world_model = train_world_model_only(
        train_env,
        num_episodes=args.episodes // 2,
        verbose=True
    )
    
    # Phase 2: Full agent
    agent = train_full_agent(
        train_env,
        num_episodes=args.episodes,
        use_planning=not args.no_planning,
        verbose=True
    )
    
    # Phase 3: Evaluation
    final_metrics = run_evaluation(
        agent, train_env,
        num_episodes=args.eval_episodes,
        test_generalization=True,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    checkpoint_path = os.path.join(args.save_path, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    agent.save(checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {training_time / 60:.1f} minutes")
    print(f"\nComparison:")
    print(f"  Random Agent:     Survival {random_metrics.survival_rate:.1%}, Goals {random_metrics.goal_completion_rate:.1%}")
    print(f"  Reactive Agent:   Survival {reactive_metrics.survival_rate:.1%}, Goals {reactive_metrics.goal_completion_rate:.1%}")
    print(f"  Transformer Agent: Survival {final_metrics.survival_rate:.1%}, Goals {final_metrics.goal_completion_rate:.1%}")
    print(f"\nImprovement over random: {(final_metrics.survival_rate - random_metrics.survival_rate) / max(0.01, random_metrics.survival_rate):.1%}")
    print()


if __name__ == "__main__":
    main()
