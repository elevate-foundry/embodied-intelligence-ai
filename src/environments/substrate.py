"""
Braille Substrate - 8-bit encoding for all perception and action.

The substrate constrains all cognition to pass through a fixed discrete representation.
No privileged access to continuous latent states.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Standard 6-dot Braille patterns (dots 1-6)
# Each character maps to a unique 6-bit pattern
BRAILLE_PATTERNS = {
    'a': 0b000001, 'b': 0b000011, 'c': 0b001001, 'd': 0b011001, 'e': 0b010001,
    'f': 0b001011, 'g': 0b011011, 'h': 0b010011, 'i': 0b001010, 'j': 0b011010,
    'k': 0b000101, 'l': 0b000111, 'm': 0b001101, 'n': 0b011101, 'o': 0b010101,
    'p': 0b001111, 'q': 0b011111, 'r': 0b010111, 's': 0b001110, 't': 0b011110,
    'u': 0b100101, 'v': 0b100111, 'w': 0b111010, 'x': 0b101101, 'y': 0b111101,
    'z': 0b110101,
    # Special tokens for world state
    'empty': 0b000000,      # Empty cell
    'wall': 0b111111,       # Impassable
    'agent': 0b101010,      # Self
    'goal': 0b010101,       # Target
    'hazard': 0b110011,     # Danger
    'energy': 0b001100,     # Resource
}

# Reverse mapping for decoding
PATTERN_TO_SYMBOL = {v: k for k, v in BRAILLE_PATTERNS.items()}


@dataclass
class BrailleToken:
    """An 8-bit token: 6 bits for braille pattern, 2 bits for metadata."""
    pattern: int  # 6-bit braille pattern
    meta: int     # 2-bit metadata (intensity, state, etc.)
    
    def __post_init__(self):
        assert 0 <= self.pattern < 64, "Pattern must be 6-bit"
        assert 0 <= self.meta < 4, "Meta must be 2-bit"
    
    @property
    def value(self) -> int:
        """Full 8-bit value."""
        return (self.meta << 6) | self.pattern
    
    @classmethod
    def from_value(cls, value: int) -> 'BrailleToken':
        """Decode from 8-bit value."""
        assert 0 <= value < 256, "Value must be 8-bit"
        return cls(pattern=value & 0x3F, meta=(value >> 6) & 0x03)
    
    def __repr__(self) -> str:
        symbol = PATTERN_TO_SYMBOL.get(self.pattern, '?')
        return f"BrailleToken({symbol}, meta={self.meta})"


class BrailleSubstrate:
    """
    The substrate through which all perception and action must pass.
    
    Enforces:
    - Fixed 8-bit token vocabulary
    - Quantized sensor readings
    - Limited bandwidth (max tokens per observation)
    """
    
    def __init__(self, max_observation_tokens: int = 64, noise_level: float = 0.0):
        self.max_observation_tokens = max_observation_tokens
        self.noise_level = noise_level
        self.vocab_size = 256  # 8-bit tokens
        
    def encode_cell(self, cell_type: str, intensity: int = 0) -> BrailleToken:
        """Encode a world cell as a braille token."""
        pattern = BRAILLE_PATTERNS.get(cell_type, BRAILLE_PATTERNS['empty'])
        meta = min(intensity, 3)  # Clamp to 2 bits
        return BrailleToken(pattern=pattern, meta=meta)
    
    def decode_token(self, token: BrailleToken) -> Tuple[str, int]:
        """Decode a braille token to cell type and intensity."""
        symbol = PATTERN_TO_SYMBOL.get(token.pattern, 'empty')
        return symbol, token.meta
    
    def encode_observation(self, grid: np.ndarray, agent_pos: Tuple[int, int], 
                          view_radius: int = 3) -> np.ndarray:
        """
        Encode a partial observation of the world as braille tokens.
        
        The agent can only see within view_radius cells.
        Returns a flattened array of 8-bit token values.
        """
        h, w = grid.shape
        ax, ay = agent_pos
        
        tokens = []
        for dy in range(-view_radius, view_radius + 1):
            for dx in range(-view_radius, view_radius + 1):
                x, y = ax + dx, ay + dy
                if 0 <= x < w and 0 <= y < h:
                    cell_value = grid[y, x]
                    token = BrailleToken.from_value(int(cell_value))
                else:
                    # Out of bounds = wall
                    token = self.encode_cell('wall')
                tokens.append(token.value)
        
        # Apply noise if configured
        if self.noise_level > 0:
            tokens = self._apply_noise(tokens)
        
        # Truncate to max observation size
        tokens = tokens[:self.max_observation_tokens]
        
        return np.array(tokens, dtype=np.uint8)
    
    def _apply_noise(self, tokens: List[int]) -> List[int]:
        """Corrupt tokens with probability noise_level."""
        noisy = []
        for t in tokens:
            if np.random.random() < self.noise_level:
                # Flip random bits
                flip_mask = np.random.randint(0, 256)
                t = t ^ flip_mask
            noisy.append(t)
        return noisy
    
    def encode_action(self, action: str) -> int:
        """Encode an action as a token value."""
        action_map = {
            'up': 0, 'down': 1, 'left': 2, 'right': 3,
            'wait': 4, 'interact': 5
        }
        return action_map.get(action, 4)  # Default to wait
    
    def decode_action(self, action_id: int) -> str:
        """Decode an action token to action string."""
        actions = ['up', 'down', 'left', 'right', 'wait', 'interact']
        return actions[action_id % len(actions)]
