#!/usr/bin/env python3
"""
Embodied Intelligence AI - Demo Script

Demonstrates the core loop:
    observe → decide → act → world changes → observe

Run with: python run_demo.py
"""

import numpy as np
from src.environments import BrailleGridWorld
from src.agents.base import RandomAgent, ReactiveAgent
from src.models.world_model import WorldModel
from src.training.loop import TrainingLoop


def run_single_episode_demo():
    """Run a single episode with visualization."""
    print("=" * 60)
    print("EMBODIED INTELLIGENCE AI - Single Episode Demo")
    print("=" * 60)
    print()
    
    # Create environment
    env = BrailleGridWorld(
        width=10,
        height=10,
        view_radius=2,
        energy_cost_per_step=1,
        hazard_damage=20,
        stochastic=False,
        seed=42
    )
    
    # Create agent
    observation_size = (2 * env.view_radius + 1) ** 2
    agent = ReactiveAgent(
        observation_size=observation_size,
        action_size=6,
        seed=42
    )
    
    # Reset environment
    observation = env.reset()
    agent.reset_episode()
    
    print("Initial World State (debug view - agent cannot see this):")
    print(env.render_ascii())
    print()
    print(f"Agent observation (braille tokens): {observation[:10]}...")
    print()
    
    # Run episode
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 50:
        # Agent acts
        action = agent.act(observation)
        action_names = ['up', 'down', 'left', 'right', 'wait', 'interact']
        
        # Environment responds
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Agent learns
        agent.learn(observation, action, reward, next_obs, done)
        agent.step()
        
        total_reward += reward
        step += 1
        
        # Log significant events
        if info.get('goal_reached'):
            print(f"Step {step}: GOAL REACHED! Reward: {reward:.2f}")
        elif info.get('hazard'):
            print(f"Step {step}: Hit hazard! Energy: {env.agent.energy}")
        elif info.get('energy_collected'):
            print(f"Step {step}: Collected energy! Energy: {env.agent.energy}")
        
        observation = next_obs
    
    print()
    print("Final World State:")
    print(env.render_ascii())
    print()
    print(f"Episode Summary:")
    print(f"  Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Survived: {env.agent.alive}")
    print(f"  Termination: {info.get('reason', 'max_steps')}")
    print()
    
    return total_reward


def run_training_demo():
    """Run a short training session."""
    print("=" * 60)
    print("EMBODIED INTELLIGENCE AI - Training Demo")
    print("=" * 60)
    print()
    
    # Create environment
    env = BrailleGridWorld(
        width=8,
        height=8,
        view_radius=2,
        energy_cost_per_step=1,
        hazard_damage=15,
        stochastic=True,
        noise_level=0.02,
        seed=123
    )
    
    # Create agent
    observation_size = (2 * env.view_radius + 1) ** 2
    agent = ReactiveAgent(
        observation_size=observation_size,
        action_size=6,
        memory_capacity=16,
        seed=123
    )
    
    # Create world model
    world_model = WorldModel()
    
    # Create training loop
    trainer = TrainingLoop(
        environment=env,
        agent=agent,
        world_model=world_model,
        max_episodes=100,
        max_steps_per_episode=200,
        log_interval=20,
        seed=123
    )
    
    print("Training reactive agent for 100 episodes...")
    print("The agent learns from consequences, not explanations.")
    print()
    
    # Run training
    stats = trainer.run(verbose=True)
    
    print()
    print("Training Complete!")
    print("-" * 40)
    summary = trainer.get_summary()
    print(f"Total Episodes: {summary['total_episodes']}")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Survival Rate: {summary['survival_rate']:.1%}")
    print(f"Goal Completion Rate: {summary['goal_completion_rate']:.1%}")
    print(f"Average Episode Length: {summary['average_episode_length']:.1f}")
    print()
    
    # Memory statistics
    mem_stats = summary['agent_stats']['memory']
    print("Memory Statistics:")
    print(f"  Active Slots: {mem_stats['active_slots']}/{mem_stats['capacity']}")
    print(f"  Corrupted Slots: {mem_stats['corrupted_slots']}")
    print(f"  Total Writes: {mem_stats['total_writes']}")
    print(f"  Total Reads: {mem_stats['total_reads']}")
    print()
    
    return summary


def main():
    """Run all demos."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          EMBODIED INTELLIGENCE AI                         ║")
    print("║                                                            ║")
    print("║  Intelligence that emerges from surviving in a world      ║")
    print("║  with consequences.                                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Run single episode demo
    run_single_episode_demo()
    
    print()
    print("-" * 60)
    print()
    
    # Run training demo
    run_training_demo()
    
    print()
    print("Demo complete. See README.md for project roadmap.")
    print()


if __name__ == "__main__":
    main()
