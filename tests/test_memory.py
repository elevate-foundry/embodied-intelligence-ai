"""Tests for the external memory system."""

import numpy as np
import pytest
from src.memory.external_memory import ExternalMemory, MemorySlot


class TestMemorySlot:
    """Tests for individual memory slots."""
    
    def test_degradation(self):
        """Test memory strength degradation."""
        content = np.array([1, 2, 3], dtype=np.uint8)
        slot = MemorySlot(content=content, timestamp=0, strength=1.0)
        
        initial_strength = slot.strength
        slot.degrade(rate=0.1)
        
        assert slot.strength < initial_strength
        assert slot.strength == 0.9
    
    def test_corruption(self):
        """Test memory corruption."""
        content = np.zeros(10, dtype=np.uint8)
        slot = MemorySlot(content=content, timestamp=0, strength=1.0)
        
        rng = np.random.default_rng(42)
        slot.corrupt(noise_level=1.0, rng=rng)  # 100% corruption
        
        assert slot.corrupted
        assert not np.all(slot.content == 0)


class TestExternalMemory:
    """Tests for the external memory system."""
    
    def test_write_read(self):
        """Test basic write and read operations."""
        memory = ExternalMemory(capacity=8, slot_size=16, seed=42)
        
        content = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        success = memory.write(0, content)
        assert success
        
        retrieved, strength = memory.read(0)
        assert retrieved is not None
        assert np.array_equal(retrieved[:5], content)
        assert strength == 1.0
    
    def test_overwrite(self):
        """Test that writing overwrites previous content."""
        memory = ExternalMemory(capacity=8, slot_size=16, seed=42)
        
        content1 = np.array([1, 1, 1], dtype=np.uint8)
        content2 = np.array([2, 2, 2], dtype=np.uint8)
        
        memory.write(0, content1)
        memory.write(0, content2)
        
        retrieved, _ = memory.read(0)
        assert np.array_equal(retrieved[:3], content2)
    
    def test_degradation_over_time(self):
        """Test that memories degrade over time."""
        memory = ExternalMemory(
            capacity=8, 
            slot_size=16, 
            degradation_rate=0.1,
            seed=42
        )
        
        content = np.array([1, 2, 3], dtype=np.uint8)
        memory.write(0, content)
        
        initial_strength = memory.slots[0].strength
        
        for _ in range(10):
            memory.step()
        
        final_strength = memory.slots[0].strength
        assert final_strength < initial_strength
    
    def test_capacity_limit(self):
        """Test that memory respects capacity limits."""
        memory = ExternalMemory(capacity=4, slot_size=8, seed=42)
        
        # Try to write beyond capacity
        success = memory.write(10, np.array([1, 2, 3], dtype=np.uint8))
        assert not success
    
    def test_stats(self):
        """Test memory statistics."""
        memory = ExternalMemory(capacity=8, slot_size=16, seed=42)
        
        memory.write(0, np.array([1], dtype=np.uint8))
        memory.write(1, np.array([2], dtype=np.uint8))
        memory.read(0)
        
        stats = memory.stats()
        assert stats['active_slots'] == 2
        assert stats['total_writes'] == 2
        assert stats['total_reads'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
