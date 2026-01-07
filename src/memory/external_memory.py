"""
External Memory with Corruption and Degradation.

Persistent, fallible memory that can:
- Degrade over time
- Be overwritten
- Mislead future behavior

No magical recurrence - all memory is explicit and external.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class MemorySlot:
    """A single memory slot with degradation properties."""
    content: np.ndarray  # The stored tokens
    timestamp: int       # When it was written
    strength: float      # Retrieval strength (degrades over time)
    corrupted: bool = False
    
    def degrade(self, rate: float = 0.01):
        """Reduce memory strength over time."""
        self.strength = max(0.0, self.strength - rate)
        
    def corrupt(self, noise_level: float = 0.1, rng: np.random.Generator = None):
        """Apply random corruption to stored content."""
        if rng is None:
            rng = np.random.default_rng()
        
        mask = rng.random(self.content.shape) < noise_level
        noise = rng.integers(0, 256, self.content.shape, dtype=np.uint8)
        self.content = np.where(mask, noise, self.content)
        self.corrupted = True


class ExternalMemory:
    """
    External memory bank for the agent.
    
    Properties:
    - Fixed capacity (no infinite storage)
    - Explicit read/write operations
    - Degradation over time
    - Corruption under stress
    - No hidden state - everything is in the tokens
    """
    
    def __init__(
        self,
        capacity: int = 32,
        slot_size: int = 64,
        degradation_rate: float = 0.005,
        corruption_probability: float = 0.01,
        seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.slot_size = slot_size
        self.degradation_rate = degradation_rate
        self.corruption_probability = corruption_probability
        
        self.rng = np.random.default_rng(seed)
        self.slots: List[Optional[MemorySlot]] = [None] * capacity
        self.current_time = 0
        self.write_count = 0
        self.read_count = 0
        
    def write(self, address: int, content: np.ndarray) -> bool:
        """
        Write content to a memory slot.
        
        Writing overwrites previous content - this is irreversible.
        Returns True if write succeeded.
        """
        if address < 0 or address >= self.capacity:
            return False
        
        # Pad or truncate content to slot size
        if len(content) < self.slot_size:
            padded = np.zeros(self.slot_size, dtype=np.uint8)
            padded[:len(content)] = content
            content = padded
        else:
            content = content[:self.slot_size].copy()
        
        # Overwrite existing memory (irreversible)
        self.slots[address] = MemorySlot(
            content=content,
            timestamp=self.current_time,
            strength=1.0
        )
        
        self.write_count += 1
        return True
    
    def read(self, address: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Read content from a memory slot.
        
        Returns (content, strength) or (None, 0.0) if empty.
        Reading may trigger corruption based on probability.
        """
        if address < 0 or address >= self.capacity:
            return None, 0.0
        
        slot = self.slots[address]
        if slot is None:
            return None, 0.0
        
        self.read_count += 1
        
        # Reading can cause corruption (simulates unreliable retrieval)
        if self.rng.random() < self.corruption_probability:
            slot.corrupt(noise_level=0.05, rng=self.rng)
        
        return slot.content.copy(), slot.strength
    
    def step(self):
        """
        Advance time and apply degradation to all memories.
        
        Called once per environment step.
        """
        self.current_time += 1
        
        for slot in self.slots:
            if slot is not None:
                slot.degrade(self.degradation_rate)
                
                # Random corruption can occur over time
                if self.rng.random() < self.corruption_probability * 0.1:
                    slot.corrupt(noise_level=0.02, rng=self.rng)
    
    def get_active_slots(self) -> List[int]:
        """Return indices of non-empty slots."""
        return [i for i, slot in enumerate(self.slots) if slot is not None]
    
    def get_strongest_memories(self, k: int = 5) -> List[Tuple[int, float]]:
        """Return the k strongest memories as (address, strength) pairs."""
        memories = []
        for i, slot in enumerate(self.slots):
            if slot is not None:
                memories.append((i, slot.strength))
        
        memories.sort(key=lambda x: x[1], reverse=True)
        return memories[:k]
    
    def clear(self, address: int) -> bool:
        """Clear a memory slot. This is irreversible."""
        if 0 <= address < self.capacity:
            self.slots[address] = None
            return True
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get a flat representation of memory state for the agent.
        
        This is what the agent actually sees - not the raw slots.
        """
        state = np.zeros((self.capacity, self.slot_size + 2), dtype=np.float32)
        
        for i, slot in enumerate(self.slots):
            if slot is not None:
                state[i, :self.slot_size] = slot.content / 255.0  # Normalize
                state[i, -2] = slot.strength
                state[i, -1] = 1.0 if slot.corrupted else 0.0
        
        return state.flatten()
    
    def stats(self) -> dict:
        """Return memory usage statistics."""
        active = sum(1 for s in self.slots if s is not None)
        corrupted = sum(1 for s in self.slots if s is not None and s.corrupted)
        avg_strength = np.mean([s.strength for s in self.slots if s is not None]) if active > 0 else 0.0
        
        return {
            'active_slots': active,
            'capacity': self.capacity,
            'utilization': active / self.capacity,
            'corrupted_slots': corrupted,
            'average_strength': avg_strength,
            'total_writes': self.write_count,
            'total_reads': self.read_count,
            'current_time': self.current_time
        }
