"""
Braille-Encoded Gridworld with Irreversible Dynamics.

A minimal environment that enforces:
- Partial observability (agent sees local neighborhood only)
- Irreversible actions (energy depletes, walls can be destroyed, no resets)
- Consequence-driven learning (mistakes have permanent effects)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from .substrate import BrailleSubstrate, BRAILLE_PATTERNS


@dataclass
class AgentState:
    """The agent's internal state - subject to the body's constraints."""
    x: int
    y: int
    energy: int = 100
    alive: bool = True
    steps_taken: int = 0
    
    def deplete_energy(self, amount: int = 1):
        """Energy depletion is irreversible."""
        self.energy = max(0, self.energy - amount)
        if self.energy <= 0:
            self.alive = False


@dataclass 
class WorldState:
    """Hidden world state - agent has no direct access."""
    grid: np.ndarray
    goals_remaining: int = 0
    hazards_triggered: List[Tuple[int, int]] = field(default_factory=list)
    permanent_changes: List[str] = field(default_factory=list)


class BrailleGridWorld:
    """
    A gridworld where all perception is braille-encoded.
    
    Key properties:
    - Hidden state: Agent cannot inspect world directly
    - Partial observability: Limited view radius
    - Irreversibility: Energy depletes, world changes persist
    - No free resets: Episode ends on death or goal completion
    """
    
    # Cell type IDs (stored in grid)
    EMPTY = BRAILLE_PATTERNS['empty']
    WALL = BRAILLE_PATTERNS['wall']
    AGENT = BRAILLE_PATTERNS['agent']
    GOAL = BRAILLE_PATTERNS['goal']
    HAZARD = BRAILLE_PATTERNS['hazard']
    ENERGY = BRAILLE_PATTERNS['energy']
    
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        view_radius: int = 2,
        energy_cost_per_step: int = 1,
        hazard_damage: int = 20,
        stochastic: bool = False,
        noise_level: float = 0.0,
        seed: Optional[int] = None
    ):
        self.width = width
        self.height = height
        self.view_radius = view_radius
        self.energy_cost_per_step = energy_cost_per_step
        self.hazard_damage = hazard_damage
        self.stochastic = stochastic
        
        self.substrate = BrailleSubstrate(
            max_observation_tokens=(2 * view_radius + 1) ** 2,
            noise_level=noise_level
        )
        
        self.rng = np.random.default_rng(seed)
        self.agent: Optional[AgentState] = None
        self.world: Optional[WorldState] = None
        self._episode_count = 0
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Initialize a new episode.
        
        Note: This is the ONLY reset allowed. Once an episode starts,
        there are no checkpoints or saves.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self._episode_count += 1
        
        # Create world grid
        grid = np.full((self.height, self.width), self.EMPTY, dtype=np.uint8)
        
        # Add walls around perimeter
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL
        
        # Add random internal walls
        n_walls = self.rng.integers(3, 8)
        for _ in range(n_walls):
            x = self.rng.integers(2, self.width - 2)
            y = self.rng.integers(2, self.height - 2)
            grid[y, x] = self.WALL
        
        # Place hazards
        n_hazards = self.rng.integers(2, 5)
        for _ in range(n_hazards):
            x, y = self._find_empty_cell(grid)
            grid[y, x] = self.HAZARD
        
        # Place energy pickups
        n_energy = self.rng.integers(2, 4)
        for _ in range(n_energy):
            x, y = self._find_empty_cell(grid)
            grid[y, x] = self.ENERGY
        
        # Place goal
        gx, gy = self._find_empty_cell(grid)
        grid[gy, gx] = self.GOAL
        
        # Place agent
        ax, ay = self._find_empty_cell(grid)
        
        # Initialize states
        self.agent = AgentState(x=ax, y=ay, energy=100)
        self.world = WorldState(grid=grid, goals_remaining=1)
        
        return self._get_observation()
    
    def _find_empty_cell(self, grid: np.ndarray) -> Tuple[int, int]:
        """Find a random empty cell."""
        while True:
            x = self.rng.integers(1, self.width - 1)
            y = self.rng.integers(1, self.height - 1)
            if grid[y, x] == self.EMPTY:
                return x, y
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return (observation, reward, terminated, truncated, info).
        
        Actions: 0=up, 1=down, 2=left, 3=right, 4=wait, 5=interact
        """
        if not self.agent.alive:
            return self._get_observation(), 0.0, True, False, {"reason": "already_dead"}
        
        reward = 0.0
        info = {}
        
        # Decode action
        action_name = self.substrate.decode_action(action)
        
        # Calculate intended new position
        dx, dy = 0, 0
        if action_name == 'up':
            dy = -1
        elif action_name == 'down':
            dy = 1
        elif action_name == 'left':
            dx = -1
        elif action_name == 'right':
            dx = 1
        
        new_x = self.agent.x + dx
        new_y = self.agent.y + dy
        
        # Apply stochastic noise to movement
        if self.stochastic and (dx != 0 or dy != 0):
            if self.rng.random() < 0.1:
                # 10% chance of slipping perpendicular
                if dx != 0:
                    new_y += self.rng.choice([-1, 1])
                else:
                    new_x += self.rng.choice([-1, 1])
        
        # Check collision
        target_cell = self.world.grid[new_y, new_x]
        
        if target_cell == self.WALL:
            # Bump into wall - no movement, still costs energy
            info['collision'] = True
            reward -= 0.1
        else:
            # Move to new position
            self.agent.x = new_x
            self.agent.y = new_y
            
            # Handle cell interactions
            if target_cell == self.HAZARD:
                self.agent.deplete_energy(self.hazard_damage)
                self.world.hazards_triggered.append((new_x, new_y))
                self.world.permanent_changes.append(f"hazard_triggered_{new_x}_{new_y}")
                reward -= 1.0
                info['hazard'] = True
                
            elif target_cell == self.GOAL:
                self.world.goals_remaining -= 1
                self.world.grid[new_y, new_x] = self.EMPTY
                reward += 10.0
                info['goal_reached'] = True
                
            elif target_cell == self.ENERGY:
                self.agent.energy = min(100, self.agent.energy + 30)
                self.world.grid[new_y, new_x] = self.EMPTY  # Consumed - irreversible
                reward += 0.5
                info['energy_collected'] = True
        
        # Energy cost for existing
        self.agent.deplete_energy(self.energy_cost_per_step)
        self.agent.steps_taken += 1
        
        # Small penalty per step to encourage efficiency
        reward -= 0.01
        
        # Check termination conditions
        terminated = False
        if not self.agent.alive:
            terminated = True
            reward -= 5.0
            info['reason'] = 'energy_depleted'
        elif self.world.goals_remaining == 0:
            terminated = True
            info['reason'] = 'goal_completed'
        
        truncated = self.agent.steps_taken >= 500
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the agent's partial observation through the braille substrate.
        
        The agent ONLY sees through this interface - no direct grid access.
        """
        return self.substrate.encode_observation(
            self.world.grid,
            (self.agent.x, self.agent.y),
            self.view_radius
        )
    
    def get_info(self) -> Dict:
        """Return non-privileged info about agent state."""
        return {
            'energy': self.agent.energy,
            'steps': self.agent.steps_taken,
            'alive': self.agent.alive,
            'episode': self._episode_count
        }
    
    def render_ascii(self) -> str:
        """Debug rendering - NOT available to agent."""
        symbols = {
            self.EMPTY: '.',
            self.WALL: '#',
            self.GOAL: 'G',
            self.HAZARD: 'X',
            self.ENERGY: 'E',
        }
        
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if x == self.agent.x and y == self.agent.y:
                    row.append('@')
                else:
                    cell = self.world.grid[y, x]
                    row.append(symbols.get(cell, '?'))
            lines.append(''.join(row))
        
        lines.append(f"Energy: {self.agent.energy} | Steps: {self.agent.steps_taken}")
        return '\n'.join(lines)
