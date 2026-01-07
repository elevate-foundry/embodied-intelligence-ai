"""
Transformer World Model - Full PyTorch Implementation.

Predicts future observations under action constraints.
Trained first to predict world dynamics, then used for planning.
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TransformerConfig:
    """Configuration for the transformer world model."""
    vocab_size: int = 256           # 8-bit braille tokens
    action_size: int = 6            # Number of possible actions
    observation_size: int = 25      # Tokens per observation (5x5 grid)
    hidden_dim: int = 128           # Transformer hidden dimension
    num_layers: int = 4             # Number of transformer layers
    num_heads: int = 4              # Number of attention heads
    ff_dim: int = 512               # Feed-forward dimension
    max_seq_len: int = 256          # Maximum sequence length
    dropout: float = 0.1
    

if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""
        
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    
    class TransformerWorldModel(nn.Module):
        """
        Transformer-based world model for embodied intelligence.
        
        Input: Sequence of (observation, action) pairs
        Output: Predicted next observation tokens
        
        The model learns to predict world dynamics from experience,
        enabling the agent to anticipate consequences of actions.
        """
        
        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.config = config
            
            # Token embeddings
            self.obs_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
            self.action_embedding = nn.Embedding(config.action_size, config.hidden_dim)
            
            # Positional encoding
            self.pos_encoding = PositionalEncoding(
                config.hidden_dim, 
                config.max_seq_len, 
                config.dropout
            )
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
            
            # Output projection - predict next observation tokens
            self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
            
            # Layer norm
            self.ln = nn.LayerNorm(config.hidden_dim)
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights with small values."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.1)
        
        def forward(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                observations: (batch, seq_len, obs_size) observation tokens
                actions: (batch, seq_len) action indices
                mask: Optional attention mask
                
            Returns:
                logits: (batch, seq_len, obs_size, vocab_size) predicted token logits
            """
            batch_size, seq_len, obs_size = observations.shape
            
            # Embed observations: (batch, seq_len, obs_size, hidden_dim)
            obs_emb = self.obs_embedding(observations)
            
            # Embed actions: (batch, seq_len, hidden_dim)
            act_emb = self.action_embedding(actions)
            
            # Combine: flatten obs tokens and interleave with actions
            # Each timestep: [obs_token_1, ..., obs_token_n, action]
            tokens_per_step = obs_size + 1
            total_tokens = seq_len * tokens_per_step
            
            # Reshape observations: (batch, seq_len * obs_size, hidden_dim)
            obs_flat = obs_emb.view(batch_size, seq_len * obs_size, -1)
            
            # Create combined sequence
            combined = torch.zeros(
                batch_size, total_tokens, self.config.hidden_dim,
                device=observations.device, dtype=obs_emb.dtype
            )
            
            for t in range(seq_len):
                start_idx = t * tokens_per_step
                # Insert observation tokens
                combined[:, start_idx:start_idx + obs_size] = obs_emb[:, t]
                # Insert action token
                combined[:, start_idx + obs_size] = act_emb[:, t]
            
            # Add positional encoding
            combined = self.pos_encoding(combined)
            
            # Create causal mask
            causal_mask = self._generate_causal_mask(total_tokens, observations.device)
            
            # Transformer forward
            hidden = self.transformer(combined, mask=causal_mask)
            hidden = self.ln(hidden)
            
            # Project to vocabulary
            logits = self.output_proj(hidden)
            
            # Reshape to (batch, seq_len, obs_size + 1, vocab_size)
            logits = logits.view(batch_size, seq_len, tokens_per_step, -1)
            
            # We only care about predicting the next observation tokens
            # Return predictions for observation positions only
            return logits[:, :, :obs_size, :]
        
        def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
            """Generate causal attention mask."""
            mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            return mask
        
        def predict_next(
            self,
            observation: torch.Tensor,
            action: torch.Tensor,
            context_obs: Optional[torch.Tensor] = None,
            context_act: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Predict the next observation given current state and action.
            
            Args:
                observation: (batch, obs_size) current observation
                action: (batch,) action to take
                context_obs: Optional (batch, ctx_len, obs_size) history
                context_act: Optional (batch, ctx_len) action history
                
            Returns:
                predicted: (batch, obs_size) predicted next observation tokens
            """
            batch_size = observation.shape[0]
            
            if context_obs is not None and context_act is not None:
                # Append current to context
                observations = torch.cat([
                    context_obs, 
                    observation.unsqueeze(1)
                ], dim=1)
                actions = torch.cat([
                    context_act,
                    action.unsqueeze(1)
                ], dim=1)
            else:
                observations = observation.unsqueeze(1)
                actions = action.unsqueeze(1)
            
            # Forward pass
            logits = self.forward(observations, actions)
            
            # Get last timestep predictions
            last_logits = logits[:, -1]  # (batch, obs_size, vocab_size)
            
            # Argmax to get predicted tokens
            predicted = torch.argmax(last_logits, dim=-1)
            
            return predicted
        
        def compute_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute prediction loss.
            
            Args:
                observations: (batch, seq_len, obs_size) input observations
                actions: (batch, seq_len) actions taken
                next_observations: (batch, seq_len, obs_size) target observations
                
            Returns:
                loss: Cross-entropy loss for next-token prediction
            """
            logits = self.forward(observations, actions)
            
            # Reshape for cross-entropy (use reshape instead of view for non-contiguous tensors)
            batch_size, seq_len, obs_size, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = next_observations.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat.long())
            return loss
    
    
    class TransitionDataset(Dataset):
        """Dataset of (observation, action, next_observation) transitions."""
        
        def __init__(
            self,
            observations: np.ndarray,
            actions: np.ndarray,
            next_observations: np.ndarray
        ):
            self.observations = torch.from_numpy(observations).long()
            self.actions = torch.from_numpy(actions).long()
            self.next_observations = torch.from_numpy(next_observations).long()
        
        def __len__(self):
            return len(self.observations)
        
        def __getitem__(self, idx):
            return (
                self.observations[idx],
                self.actions[idx],
                self.next_observations[idx]
            )
    
    
    class SequenceDataset(Dataset):
        """Dataset of observation-action sequences for training."""
        
        def __init__(
            self,
            episodes: List[Tuple[np.ndarray, np.ndarray]],
            seq_len: int = 16
        ):
            """
            Args:
                episodes: List of (observations, actions) arrays per episode
                seq_len: Sequence length for training
            """
            self.sequences = []
            self.seq_len = seq_len
            
            for obs_seq, act_seq in episodes:
                # Create overlapping sequences
                for i in range(len(obs_seq) - seq_len):
                    self.sequences.append((
                        obs_seq[i:i + seq_len],
                        act_seq[i:i + seq_len],
                        obs_seq[i + 1:i + seq_len + 1]  # Next observations
                    ))
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            obs, act, next_obs = self.sequences[idx]
            return (
                torch.from_numpy(obs).long(),
                torch.from_numpy(act).long(),
                torch.from_numpy(next_obs).long()
            )


class WorldModelTrainer:
    """
    Trainer for the transformer world model.
    
    Implements:
    - Online learning from environment interactions
    - Batch training from replay buffer
    - Evaluation on held-out transitions
    """
    
    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = "auto"
    ):
        self.config = config or TransformerConfig()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for WorldModelTrainer")
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create model
        self.model = TransformerWorldModel(self.config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Replay buffer
        self.replay_buffer: List[Tuple[np.ndarray, int, np.ndarray]] = []
        self.max_buffer_size = 100000
        
        # Training stats
        self.train_steps = 0
        self.total_loss = 0.0
    
    def add_transition(
        self,
        observation: np.ndarray,
        action: int,
        next_observation: np.ndarray
    ):
        """Add a transition to the replay buffer."""
        self.replay_buffer.append((observation, action, next_observation))
        
        # Limit buffer size
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size // 2:]
    
    def train_step(self) -> float:
        """Perform one training step on a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        
        obs_batch = []
        act_batch = []
        next_obs_batch = []
        
        for idx in indices:
            obs, act, next_obs = self.replay_buffer[idx]
            obs_batch.append(obs)
            act_batch.append(act)
            next_obs_batch.append(next_obs)
        
        # Convert to tensors
        observations = torch.from_numpy(np.array(obs_batch)).long().to(self.device)
        actions = torch.from_numpy(np.array(act_batch)).long().to(self.device)
        next_observations = torch.from_numpy(np.array(next_obs_batch)).long().to(self.device)
        
        # Add sequence dimension (single step)
        observations = observations.unsqueeze(1)
        actions = actions.unsqueeze(1)
        next_observations = next_observations.unsqueeze(1)
        
        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model.compute_loss(observations, actions, next_observations)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        self.total_loss += loss.item()
        
        return loss.item()
    
    def predict(
        self,
        observation: np.ndarray,
        action: int
    ) -> np.ndarray:
        """Predict next observation."""
        self.model.eval()
        
        with torch.no_grad():
            obs = torch.from_numpy(observation).long().unsqueeze(0).to(self.device)
            act = torch.tensor([action]).long().to(self.device)
            
            predicted = self.model.predict_next(obs, act)
            
        return predicted.cpu().numpy()[0]
    
    def evaluate(
        self,
        test_transitions: List[Tuple[np.ndarray, int, np.ndarray]]
    ) -> dict:
        """Evaluate prediction accuracy on test transitions."""
        self.model.eval()
        
        correct_tokens = 0
        total_tokens = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for obs, act, next_obs in test_transitions:
                predicted = self.predict(obs, act)
                
                correct_tokens += np.sum(predicted == next_obs)
                total_tokens += len(next_obs)
                
                # Compute loss
                obs_t = torch.from_numpy(obs).long().unsqueeze(0).unsqueeze(0).to(self.device)
                act_t = torch.tensor([[act]]).long().to(self.device)
                next_t = torch.from_numpy(next_obs).long().unsqueeze(0).unsqueeze(0).to(self.device)
                
                loss = self.model.compute_loss(obs_t, act_t, next_t)
                total_loss += loss.item()
        
        return {
            'token_accuracy': correct_tokens / total_tokens if total_tokens > 0 else 0.0,
            'average_loss': total_loss / len(test_transitions) if test_transitions else 0.0,
            'num_samples': len(test_transitions)
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_steps': self.train_steps
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_steps = checkpoint['train_steps']
    
    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            'train_steps': self.train_steps,
            'average_loss': self.total_loss / max(1, self.train_steps),
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }
