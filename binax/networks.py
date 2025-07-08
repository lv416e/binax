"""Neural network architectures for bin packing reinforcement learning."""

from typing import Sequence

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from binax.types import BinPackingState, NetworkOutputs


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer for bin-item interactions."""
    
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    
    def setup(self):
        self.dense_q = nn.Dense(self.num_heads * self.head_dim)
        self.dense_k = nn.Dense(self.num_heads * self.head_dim)
        self.dense_v = nn.Dense(self.num_heads * self.head_dim)
        self.dense_out = nn.Dense(self.num_heads * self.head_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, query, key, value, mask=None, training=True):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear projections
        q = self.dense_q(query)
        k = self.dense_k(key)
        v = self.dense_v(value)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        scores = scores / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        attention_weights = nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, deterministic=not training)
        
        # Apply attention to values
        context = jnp.matmul(attention_weights, v)
        
        # Reshape and apply output projection
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.dense_out(context)
        
        return output, attention_weights


class BinItemEncoder(nn.Module):
    """Encoder for bin and item representations."""
    
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    def setup(self):
        self.bin_embedding = nn.Dense(self.hidden_dim)
        self.item_embedding = nn.Dense(self.hidden_dim)
        self.position_embedding = nn.Dense(self.hidden_dim)
        
        self.attention_layers = [
            MultiHeadAttention(
                num_heads=self.num_heads,
                head_dim=self.hidden_dim // self.num_heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]
        
        self.layer_norms = [nn.LayerNorm() for _ in range(self.num_layers)]
        self.feedforward_layers = [
            nn.Sequential([
                nn.Dense(self.hidden_dim * 4),
                nn.gelu,
                nn.Dropout(self.dropout_rate),
                nn.Dense(self.hidden_dim),
                nn.Dropout(self.dropout_rate),
            ])
            for _ in range(self.num_layers)
        ]
    
    def __call__(self, state: BinPackingState, training: bool = True):
        # Encode bin states
        bin_features = jnp.stack([
            state.bin_capacities,
            state.bin_utilization,
        ], axis=-1)
        bin_embeddings = self.bin_embedding(bin_features)
        
        # Add positional encoding for bins
        bin_positions = jnp.arange(bin_features.shape[0])[:, None]
        bin_pos_embeddings = self.position_embedding(
            nn.one_hot(bin_positions, num_classes=50)
        )
        bin_embeddings = bin_embeddings + bin_pos_embeddings
        
        # Encode current item
        current_item_size = state.item_queue[state.current_item_idx:state.current_item_idx+1]
        item_embedding = self.item_embedding(current_item_size[:, None])
        
        # Concatenate bin and item embeddings
        combined_embeddings = jnp.concatenate([bin_embeddings, item_embedding], axis=0)
        combined_embeddings = combined_embeddings[None, ...]  # Add batch dimension
        
        # Apply transformer layers
        x = combined_embeddings
        for attention, layer_norm, feedforward in zip(
            self.attention_layers, self.layer_norms, self.feedforward_layers
        ):
            # Self-attention
            attn_output, _ = attention(x, x, x, training=training)
            x = layer_norm(x + attn_output)
            
            # Feedforward
            ff_output = feedforward(x, training=training)
            x = layer_norm(x + ff_output)
        
        return x.squeeze(0)  # Remove batch dimension


class PolicyValueNetwork(nn.Module):
    """Policy-value network for bin packing using attention mechanism."""
    
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    max_bins: int = 50
    dropout_rate: float = 0.1
    
    def setup(self):
        self.encoder = BinItemEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        
        # Policy head
        self.policy_projection = nn.Dense(self.hidden_dim)
        self.policy_output = nn.Dense(self.max_bins + 1)  # +1 for new bin action
        
        # Value head
        self.value_layers = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.hidden_dim // 2),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(1),
        ])
    
    def __call__(self, state: BinPackingState, training: bool = True) -> NetworkOutputs:
        # Encode state
        encoded_state = self.encoder(state, training=training)
        
        # Policy head: focus on bin embeddings for action selection
        bin_embeddings = encoded_state[:-1]  # Exclude item embedding
        policy_features = self.policy_projection(bin_embeddings)
        
        # Add new bin option
        new_bin_feature = jnp.mean(policy_features, axis=0, keepdims=True)
        policy_features = jnp.concatenate([policy_features, new_bin_feature], axis=0)
        
        # Compute action logits
        action_logits = self.policy_output(policy_features).squeeze(-1)
        
        # Value head: use global representation
        global_representation = jnp.mean(encoded_state, axis=0)
        value = self.value_layers(global_representation, training=training).squeeze(-1)
        
        return NetworkOutputs(action_logits=action_logits, value=value)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    hidden_dim: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.layers = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.hidden_dim),
        ])
        self.layer_norm = nn.LayerNorm()
    
    def __call__(self, x, training: bool = True):
        residual = x
        x = self.layers(x, training=training)
        return self.layer_norm(x + residual)


class SimplePolicyValueNetwork(nn.Module):
    """Simpler policy-value network without attention for baseline."""
    
    hidden_dims: Sequence[int] = (512, 256, 128)
    max_bins: int = 50
    dropout_rate: float = 0.1
    
    def setup(self):
        # Shared layers
        shared_layers = []
        for hidden_dim in self.hidden_dims:
            shared_layers.extend([
                nn.Dense(hidden_dim),
                nn.gelu,
                nn.Dropout(self.dropout_rate),
            ])
        self.shared_layers = shared_layers
        
        # Policy head
        self.policy_head = nn.Sequential([
            nn.Dense(self.hidden_dims[-1]),
            nn.gelu,
            nn.Dense(self.max_bins + 1),
        ])
        
        # Value head
        self.value_head = nn.Sequential([
            nn.Dense(self.hidden_dims[-1]),
            nn.gelu,
            nn.Dense(1),
        ])
    
    def __call__(self, state: BinPackingState, training: bool = True) -> NetworkOutputs:
        # Flatten state representation
        current_item_size = state.item_queue[state.current_item_idx]
        
        features = jnp.concatenate([
            state.bin_capacities,
            state.bin_utilization,
            jnp.array([current_item_size]),
            jnp.array([state.step_count / 100.0]),  # Normalized step count
        ])
        
        # Shared layers
        x = features
        for layer in self.shared_layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x, deterministic=not training)
            else:
                x = layer(x)
        
        # Policy and value outputs
        action_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        
        return NetworkOutputs(action_logits=action_logits, value=value)


def create_network(
    network_type: str = "attention",
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 8,
    max_bins: int = 50,
    dropout_rate: float = 0.1,
) -> nn.Module:
    """Factory function to create different network architectures.
    
    Args:
        network_type: Type of network ("attention" or "simple")
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        num_heads: Number of attention heads (for attention network)
        max_bins: Maximum number of bins
        dropout_rate: Dropout rate
        
    Returns:
        Network module
    """
    if network_type == "attention":
        return PolicyValueNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_bins=max_bins,
            dropout_rate=dropout_rate,
        )
    elif network_type == "simple":
        return SimplePolicyValueNetwork(
            hidden_dims=(hidden_dim * 2, hidden_dim, hidden_dim // 2),
            max_bins=max_bins,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def init_network_params(
    network: nn.Module,
    key: chex.PRNGKey,
    dummy_state: BinPackingState,
) -> chex.ArrayTree:
    """Initialize network parameters.
    
    Args:
        network: Network module
        key: JAX random key
        dummy_state: Dummy state for initialization
        
    Returns:
        Initialized network parameters
    """
    return network.init(key, dummy_state, training=False)