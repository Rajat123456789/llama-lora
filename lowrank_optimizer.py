"""
Low-Rank Adaptive Optimizer (GaLore-style optimization for LoRA)

Instead of just compressing weights, we compress the optimizer states
(gradients, momentum, variance) in low-rank space during training.

This reduces memory from O(m×n) to O((m+n)×r) for optimizer states.
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class LowRankAdamW(Optimizer):
    """
    AdamW optimizer that operates in low-rank subspace for memory efficiency.
    
    Key Innovation:
    - Projects gradients into low-rank space before computing momentum/variance
    - Stores optimizer states in compressed form
    - Memory: O((m+n)×r) instead of O(m×n)
    
    Args:
        params: Model parameters
        lr: Learning rate
        betas: Adam beta coefficients (β₁, β₂)
        eps: Numerical stability constant
        weight_decay: L2 penalty coefficient
        rank: Low-rank dimension (r)
        projection_update_freq: How often to update projection matrices (T)
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        rank=128,
        projection_update_freq=100,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            projection_update_freq=projection_update_freq,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            rank = group["rank"]
            proj_freq = group["projection_update_freq"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Skip small parameters or 1D tensors (no compression needed)
                if grad.dim() < 2 or grad.numel() < rank * 2:
                    # Standard Adam update for small params
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    
                    # Weight decay
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    
                    # Momentum and variance
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = lr / bias_correction1
                    
                    # Update
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.add_(exp_avg / denom, alpha=-step_size)
                    
                else:
                    # Low-rank compressed optimizer for large 2D params
                    state = self.state[p]
                    
                    # Initialize state
                    if len(state) == 0:
                        state["step"] = 0
                        m, n = grad.shape
                        r = min(rank, min(m, n) // 2)
                        
                        # Projection matrices P (m×r) and Q (n×r)
                        state["P"] = torch.randn(m, r, device=grad.device, dtype=grad.dtype) / math.sqrt(r)
                        state["Q"] = torch.randn(n, r, device=grad.device, dtype=grad.dtype) / math.sqrt(r)
                        
                        # Compressed momentum and variance (r×r)
                        state["M"] = torch.zeros(r, r, device=grad.device, dtype=grad.dtype)
                        state["V"] = torch.zeros(r, r, device=grad.device, dtype=grad.dtype)
                        state["last_proj_update"] = 0
                    
                    state["step"] += 1
                    P = state["P"]
                    Q = state["Q"]
                    M = state["M"]
                    V = state["V"]
                    
                    # Update projection matrices periodically using SVD
                    if state["step"] - state["last_proj_update"] >= proj_freq:
                        try:
                            U, S, Vh = torch.svd_lowrank(grad, q=rank)
                            state["P"] = U
                            state["Q"] = Vh.T
                            state["last_proj_update"] = state["step"]
                        except:
                            pass  # Keep current projections if SVD fails
                    
                    P = state["P"]
                    Q = state["Q"]
                    
                    # Project gradient into low-rank space: R = P^T @ G @ Q
                    R = P.T @ grad @ Q
                    
                    # Update momentum in low-rank space: M = β₁M + (1-β₁)R
                    M.mul_(beta1).add_(R, alpha=1 - beta1)
                    
                    # Update variance in low-rank space: V = β₂V + (1-β₂)R²
                    V.mul_(beta2).add_(R * R, alpha=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    
                    # Normalized update: N = M / √(V + ε)
                    N = M / (V.sqrt() / math.sqrt(bias_correction2) + eps)
                    N = N / bias_correction1
                    
                    # Project back to full space: G̃ = P @ N @ Q^T
                    G_tilde = P @ N @ Q.T
                    
                    # Weight decay
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    
                    # Update weights: W = W - η·G̃
                    p.add_(G_tilde, alpha=-lr)
        
        return loss


def get_optimizer(model, config_name="standard"):
    """
    Get optimizer based on configuration.
    
    Args:
        model: The model to optimize
        config_name: "standard" (AdamW) or "lowrank" (LowRankAdamW)
    
    Returns:
        Optimizer instance
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config_name == "lowrank":
        return LowRankAdamW(
            trainable_params,
            lr=2e-4,
            rank=128,
            projection_update_freq=100,
        )
    else:
        # Standard AdamW
        return torch.optim.AdamW(trainable_params, lr=2e-4)
