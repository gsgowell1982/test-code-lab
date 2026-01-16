"""
Mixture of Experts (MoE) Architecture - Simple Implementation

This module implements a basic MoE architecture with:
- Multiple expert networks (feedforward networks)
- A gating network that routes inputs to experts
- Top-K expert selection mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Single Expert Network - A simple feedforward network.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    """
    Gating Network - Decides which experts to use for each input.
    
    Args:
        input_dim: Input feature dimension
        num_experts: Number of expert networks
    """
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits for each expert
        return self.gate(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer.
    
    This layer routes inputs to multiple experts and combines their outputs
    using a weighted sum based on the gating network's decisions.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for each expert
        output_dim: Output feature dimension
        num_experts: Number of expert networks
        top_k: Number of experts to select for each input (default: 2)
        noise_std: Standard deviation of noise added to gating logits during training
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gating = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            output: Combined expert outputs of shape (batch_size, output_dim)
            load_balance_loss: Auxiliary loss for load balancing
        """
        batch_size = x.shape[0]
        
        # Get gating logits
        gate_logits = self.gating(x)  # (batch_size, num_experts)
        
        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Apply softmax to get weights (only over selected experts)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (batch_size, top_k)
        
        # Compute expert outputs for selected experts
        # Initialize output tensor
        output = torch.zeros(batch_size, self.experts[0].net[-1].out_features, device=x.device)
        
        # For each expert, find which samples selected it and compute weighted output
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (batch_size,)
            expert_weights = top_k_weights[:, k].unsqueeze(-1)  # (batch_size, 1)
            
            for expert_idx in range(self.num_experts):
                # Find samples that selected this expert at position k
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_weights[mask] * expert_output
        
        # Compute load balancing loss (encourages equal usage of experts)
        # Fraction of tokens routed to each expert
        gates = F.softmax(gate_logits, dim=-1)
        expert_usage = gates.mean(dim=0)  # (num_experts,)
        
        # Compute load balance loss (variance of expert usage)
        load_balance_loss = self.num_experts * (expert_usage ** 2).sum()
        
        return output, load_balance_loss


class MoETransformerBlock(nn.Module):
    """
    A Transformer block with MoE feedforward layer.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        num_experts: Number of experts in MoE layer
        top_k: Number of experts to select
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # MoE feedforward layer
        self.moe = MoELayer(d_model, d_ff, d_model, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            load_balance_loss: Auxiliary loss from MoE layer
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # MoE feedforward with residual connection
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # Flatten for MoE
        moe_output, load_balance_loss = self.moe(x_flat)
        moe_output = moe_output.view(batch_size, seq_len, d_model)
        
        x = self.norm2(x + self.dropout(moe_output))
        
        return x, load_balance_loss


def test_moe_layer():
    """Test the basic MoE layer."""
    print("=" * 60)
    print("Testing MoE Layer")
    print("=" * 60)
    
    # Configuration
    batch_size = 16
    input_dim = 256
    hidden_dim = 512
    output_dim = 256
    num_experts = 8
    top_k = 2
    
    # Create MoE layer
    moe = MoELayer(input_dim, hidden_dim, output_dim, num_experts, top_k)
    
    # Create random input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output, load_balance_loss = moe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of experts: {num_experts}")
    print(f"Top-K experts selected: {top_k}")
    print(f"Load balance loss: {load_balance_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n✓ MoE Layer test passed!\n")


def test_moe_transformer():
    """Test the MoE Transformer block."""
    print("=" * 60)
    print("Testing MoE Transformer Block")
    print("=" * 60)
    
    # Configuration
    batch_size = 8
    seq_len = 32
    d_model = 256
    num_heads = 4
    d_ff = 512
    num_experts = 4
    top_k = 2
    
    # Create MoE Transformer block
    transformer = MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, load_balance_loss = transformer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"d_model: {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Number of experts: {num_experts}")
    print(f"Load balance loss: {load_balance_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n✓ MoE Transformer test passed!\n")


def test_training_step():
    """Test a simple training step with MoE."""
    print("=" * 60)
    print("Testing MoE Training Step")
    print("=" * 60)
    
    # Configuration
    batch_size = 16
    input_dim = 128
    hidden_dim = 256
    output_dim = 64
    num_experts = 4
    top_k = 2
    num_steps = 5
    load_balance_weight = 0.01
    
    # Create model
    moe = MoELayer(input_dim, hidden_dim, output_dim, num_experts, top_k)
    
    # Create optimizer
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)
    
    # Simple training loop
    moe.train()
    for step in range(num_steps):
        # Generate random input and target
        x = torch.randn(batch_size, input_dim)
        target = torch.randn(batch_size, output_dim)
        
        # Forward pass
        output, load_balance_loss = moe(x)
        
        # Compute main loss (MSE)
        main_loss = F.mse_loss(output, target)
        
        # Total loss = main loss + load balance loss
        total_loss = main_loss + load_balance_weight * load_balance_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}/{num_steps} - Main Loss: {main_loss.item():.4f}, "
              f"LB Loss: {load_balance_loss.item():.4f}, Total: {total_loss.item():.4f}")
    
    print("\n✓ MoE Training test passed!\n")


def test_expert_selection():
    """Visualize expert selection patterns."""
    print("=" * 60)
    print("Testing Expert Selection Patterns")
    print("=" * 60)
    
    # Configuration
    batch_size = 100
    input_dim = 64
    hidden_dim = 128
    output_dim = 64
    num_experts = 8
    top_k = 2
    
    # Create MoE layer
    moe = MoELayer(input_dim, hidden_dim, output_dim, num_experts, top_k, noise_std=0.0)
    moe.eval()
    
    # Create random input
    x = torch.randn(batch_size, input_dim)
    
    # Get gating decisions
    with torch.no_grad():
        gate_logits = moe.gating(x)
        _, top_k_indices = torch.topk(gate_logits, top_k, dim=-1)
    
    # Count expert usage
    expert_counts = torch.zeros(num_experts)
    for k in range(top_k):
        for expert_idx in range(num_experts):
            expert_counts[expert_idx] += (top_k_indices[:, k] == expert_idx).sum().item()
    
    print(f"Expert usage distribution (out of {batch_size * top_k} selections):")
    for i, count in enumerate(expert_counts):
        bar = "█" * int(count / (batch_size * top_k) * 40)
        print(f"  Expert {i}: {int(count):3d} ({count / (batch_size * top_k) * 100:5.1f}%) {bar}")
    
    print("\n✓ Expert Selection test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MoE (Mixture of Experts) Architecture Test Suite")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_moe_layer()
    test_moe_transformer()
    test_training_step()
    test_expert_selection()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
