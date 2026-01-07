"""Tests for the braille gridworld environment."""

import numpy as np
import pytest
from src.environments import BrailleGridWorld, BrailleSubstrate
from src.environments.substrate import BrailleToken, BRAILLE_PATTERNS


class TestBrailleSubstrate:
    """Tests for the braille substrate encoding."""
    
    def test_token_encoding(self):
        """Test 8-bit token encoding/decoding."""
        token = BrailleToken(pattern=0b101010, meta=2)
        assert token.value == (2 << 6) | 0b101010
        
        decoded = BrailleToken.from_value(token.value)
        assert decoded.pattern == token.pattern
        assert decoded.meta == token.meta
    
    def test_cell_encoding(self):
        """Test cell type encoding."""
        substrate = BrailleSubstrate()
        
        token = substrate.encode_cell('wall', intensity=1)
        assert token.pattern == BRAILLE_PATTERNS['wall']
        assert token.meta == 1
        
        cell_type, intensity = substrate.decode_token(token)
        assert cell_type == 'wall'
        assert intensity == 1
    
    def test_action_encoding(self):
        """Test action encoding/decoding."""
        substrate = BrailleSubstrate()
        
        for action in ['up', 'down', 'left', 'right', 'wait', 'interact']:
            encoded = substrate.encode_action(action)
            decoded = substrate.decode_action(encoded)
            assert decoded == action


class TestBrailleGridWorld:
    """Tests for the gridworld environment."""
    
    def test_reset(self):
        """Test environment reset."""
        env = BrailleGridWorld(width=8, height=8, seed=42)
        obs = env.reset()
        
        assert obs is not None
        assert len(obs) == (2 * env.view_radius + 1) ** 2
        assert env.agent.alive
        assert env.agent.energy == 100
    
    def test_step(self):
        """Test environment step."""
        env = BrailleGridWorld(width=8, height=8, seed=42)
        env.reset()
        
        initial_energy = env.agent.energy
        obs, reward, terminated, truncated, info = env.step(0)  # up
        
        assert obs is not None
        assert env.agent.energy < initial_energy  # Energy depleted
        assert env.agent.steps_taken == 1
    
    def test_energy_depletion(self):
        """Test that energy depletion leads to death."""
        env = BrailleGridWorld(width=8, height=8, energy_cost_per_step=50, seed=42)
        env.reset()
        
        # Take steps until energy depletes
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(4)  # wait
            if terminated:
                break
        
        assert not env.agent.alive or env.agent.energy <= 0 or info.get('reason') == 'goal_completed'
    
    def test_partial_observability(self):
        """Test that observation size matches view radius."""
        for radius in [1, 2, 3]:
            env = BrailleGridWorld(width=10, height=10, view_radius=radius, seed=42)
            obs = env.reset()
            expected_size = (2 * radius + 1) ** 2
            assert len(obs) == expected_size
    
    def test_deterministic_seed(self):
        """Test that same seed produces same world."""
        env1 = BrailleGridWorld(width=8, height=8, seed=123)
        env2 = BrailleGridWorld(width=8, height=8, seed=123)
        
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        assert np.array_equal(obs1, obs2)
        assert env1.agent.x == env2.agent.x
        assert env1.agent.y == env2.agent.y


class TestIrreversibility:
    """Tests for irreversible dynamics."""
    
    def test_energy_pickup_consumed(self):
        """Test that energy pickups are consumed permanently."""
        env = BrailleGridWorld(width=8, height=8, seed=42)
        env.reset()
        
        # Find energy cell
        energy_positions = []
        for y in range(env.height):
            for x in range(env.width):
                if env.world.grid[y, x] == env.ENERGY:
                    energy_positions.append((x, y))
        
        initial_energy_count = len(energy_positions)
        
        # Manually move agent to energy (for testing)
        if energy_positions:
            ex, ey = energy_positions[0]
            env.agent.x = ex - 1
            env.agent.y = ey
            env.step(3)  # right
            
            # Count remaining energy
            remaining = sum(1 for y in range(env.height) for x in range(env.width)
                          if env.world.grid[y, x] == env.ENERGY)
            
            assert remaining < initial_energy_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
