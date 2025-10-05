#!/usr/bin/env python3
"""
OPT-125M Finetuning with Subspace Governance and Oracle Integration
==================================================================

This script finetunes OPT-125M on a real dataset using a sophisticated subspace governance
mechanism combined with multiple oracles for structured latent dynamics and constraints. Key features:

1. Projects hidden states into governed and free subspaces using learnable projection matrices
    with a customizable k-schedule (e.g., hourglass pattern for dimensional control across layers).
2. Integrates oracles for advanced control:
    - Energy Oracle: Applies energy-based penalties for differentiable dynamical systems.
    - CFG Oracle: Uses probabilistic context-free grammars for soft masking and loss regularization.
    - Program Oracle: Enforces hard constraints via custom token-level rules.
3. Includes anti-collapse regularizers (variance floors, log-determinant spread, radial floors)
    and orthogonality penalties to maintain subspace integrity.
4. Supports logit fusion from governed/free subspaces and dynamic scaling (e.g., beta for energy,
    lambda for CFG) with warmup schedules.
5. Uses optimizers like Muon (for spectral norm optimization) or AdamW, with full training loop,
    evaluation, and result saving.

Usage:
    python playground/finetune_opt_with_projected_latent_penalty.py --optimizer muon --dataset_size 10000
    python playground/finetune_opt_with_projected_latent_penalty.py --optimizer adam --dataset_size 5000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import numpy as np
import matplotlib
# Only use non-interactive backend if not in notebook environment
try:
    # Check if we're in a Jupyter notebook (including Colab)
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        # In Jupyter notebook - use default backend for inline plots
        pass
    else:
        # Not in notebook - use non-interactive backend
        matplotlib.use('Agg')
except (NameError, AttributeError):
    # Not in IPython environment - use non-interactive backend
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse
import time
import json
import os
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import warnings
import random
warnings.filterwarnings("ignore")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: datasets library not available. Please install it with: pip install datasets")
    exit(1)


@dataclass
class EnergyConfig:
    """Configuration for energy oracle."""
    enabled: bool = False
    mode: str = "one-step"  # "one-step" or "n-step"
    beta: float = 1.0
    horizon: int = 8
    state_dim: int = 64

@dataclass
class CFGConfig:
    """Configuration for CFG oracle."""
    enabled: bool = False
    lambda_: float = 0.1
    N: int = 8  # number of nonterminals
    nt2vocab_source: str = "all_tokens"  # or specific mapping

@dataclass
class ProgramConfig:
    """Configuration for program oracle."""
    enabled: bool = False
    constraint_type: str = "none"  # or custom constraint function

@dataclass
class GovernConfig:
    """Configuration for subspace governance."""
    k_schedule: str = "hourglass"
    k_max: int = 128
    layers: List[Union[int, str]] = None  # [2,5,8,'final'] or None for all
    final_layer_idx: int = 12  # OPT has 12 layers

@dataclass
class RegularizersConfig:
    """Configuration for anti-collapse regularizers."""
    var_tau: float = 0.5
    var_weight: float = 0.05
    logdet_delta: float = 1e-3
    logdet_weight: float = 0.01
    radial_rho: float = 0.25
    radial_weight: float = 0.02
    orth_weight: float = 1e-3

@dataclass
class FusionConfig:
    """Configuration for logit fusion."""
    scale_mask: float = 1.0

@dataclass
class TrainingConfig:
    """Configuration for training experiments with subspace governance."""
    # Model config
    model_name: str = "facebook/opt-125m"
    max_length: int = 512

    # Dataset config
    dataset_name: str = "openwebtext"
    dataset_size: int = 10000
    train_split: float = 0.8
    val_split: float = 0.2

    # Training config
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # Optimizer config
    optimizer_type: str = "muon"
    muon_lr_scale: float = 1.0
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8

    # Subspace governance config (replaces latent penalty)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    cfg: CFGConfig = field(default_factory=CFGConfig)
    program: ProgramConfig = field(default_factory=ProgramConfig)
    govern: GovernConfig = field(default_factory=GovernConfig)
    regularizers: RegularizersConfig = field(default_factory=RegularizersConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Logging config
    log_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    output_dir: str = "./opt_subspace_governance_results"

    # Experiment config
    seed: int = 42
    run_name: str = "opt_muon_subspace_governance"


class MuonOptimizer(torch.optim.Optimizer):
    """
    Muon Optimizer Implementation based on:
    "Muon Outperforms Adam in Tail-End Associative Memory Learning"

    Key innovation: replaces raw gradient with sum of normalized orthogonal factors,
    performing steepest descent with respect to the spectral norm.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Get gradient
                grad = param.grad.data

                # Muon update: replace gradient with normalized orthogonal factors
                muon_update = self._compute_muon_update(grad, param.shape)

                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    param.data.add_(param.data, alpha=-group['weight_decay'] * group['lr'])

                # Apply Muon update
                param.data.add_(muon_update, alpha=-group['lr'])

        return loss

    def _compute_muon_update(self, grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """
        Compute Muon update by replacing gradient with sum of normalized orthogonal factors.

        This is the core innovation: instead of using the raw gradient, we use
        the sum of its normalized orthogonal factors for the update.
        """
        if len(shape) == 2:  # Matrix parameter
            return self._muon_update_matrix(grad)
        elif len(shape) == 1:  # Vector parameter
            return self._muon_update_vector(grad)
        else:
            # Fallback to original gradient for other shapes
            return grad

    def _muon_update_matrix(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute Muon update for matrix parameters."""
        # SVD decomposition: grad = U * S * V^T
        U, s, Vt = torch.linalg.svd(grad, full_matrices=False)

        # Normalize singular values
        s_normalized = s / (s + 1e-8)  # Avoid division by zero

        # Reconstruct with normalized singular values
        # This is equivalent to the sum of normalized orthogonal factors
        muon_grad = U @ torch.diag(s_normalized) @ Vt

        return muon_grad

    def _muon_update_vector(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute Muon update for vector parameters."""
        # For vectors, normalization is simpler
        norm = torch.norm(grad)
        if norm > 0:
            return grad / norm
        return grad


class SubspaceGovernance(nn.Module):
    """Module for subspace governance with projection into governed and free subspaces."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.num_layers = 12  # OPT-125M has 12 layers
        self.hidden_size = 768  # OPT-125M hidden size

        # Create k-schedule for hourglass pattern
        self.k_schedule = self._create_k_schedule()
        print(f"Subspace governance k-schedule: {self.k_schedule}")

        # Per-layer projection matrices (learn U via QR orthonormalization)
        self.proj_matrices = nn.ModuleList([
            nn.Linear(self.hidden_size, k, bias=False)
            for k in self.k_schedule
        ])

        # Initialize projection matrices
        self._initialize_parameters()

    def _create_k_schedule(self) -> List[int]:
        """Create hourglass k-schedule over depth."""
        # Hourglass pattern: small → peak → small → final port
        # [32,48,64,96,128,96,64,48,32,16,16,16, 128] for final port
        base_schedule = [32, 48, 64, 96, 128, 96, 64, 48, 32, 16, 16, 16]
        final_port_k = getattr(self.config.govern, 'k_max', 128)
        return base_schedule + [final_port_k]

    def _initialize_parameters(self):
        """Initialize projection matrices with small weights."""
        for proj_matrix in self.proj_matrices:
            nn.init.normal_(proj_matrix.weight, mean=0.0, std=0.02)

    def project(self, h: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project hidden state into governed (g) and free (f) subspaces.

        Args:
            h: Hidden state tensor [batch_size, seq_len, hidden_size]
            layer_idx: Layer index for projection matrix selection

        Returns:
            g: Governed subspace projection [batch_size, seq_len, hidden_size]
            f: Free subspace residual [batch_size, seq_len, hidden_size]
            U: Projection matrix [hidden_size, k]
            gk: Governed coordinates [batch_size, seq_len, k]
        """
        U = self.proj_matrices[layer_idx].weight.T  # [hidden_size, k]

        # Project: g = (h @ U) @ U.T, f = h - g
        h_flat = h.view(-1, h.size(-1))  # [batch_size * seq_len, hidden_size]
        gk_flat = h_flat @ U             # [batch_size * seq_len, k] - coordinates
        g_flat = gk_flat @ U.T           # [batch_size * seq_len, hidden_size] - projection
        f_flat = h_flat - g_flat

        # Reshape back
        batch_size, seq_len = h.shape[:2]
        g = g_flat.view(batch_size, seq_len, -1)   # [batch_size, seq_len, hidden_size]
        f = f_flat.view(batch_size, seq_len, -1)   # [batch_size, seq_len, hidden_size]
        gk = gk_flat.view(batch_size, seq_len, -1) # [batch_size, seq_len, k]

        return g, f, U, gk

    def get_k_schedule(self) -> List[int]:
        """Get the k values for each layer."""
        return self.k_schedule

    def orth_penalty(self) -> torch.Tensor:
        """Orthogonality penalty for projection matrices."""
        loss = 0.0
        for proj_matrix in self.proj_matrices:
            U = proj_matrix.weight.T  # [d, k]
            # Orthogonality: ||U^T U - I||^2 scaled by k for stability across dimensions
            UtU = U.T @ U  # [k, k]
            k = UtU.size(0)
            I = torch.eye(k, device=UtU.device, dtype=UtU.dtype)
            loss += F.mse_loss(UtU, I, reduction='sum') / k
        return loss


def straight_through_sample(probs: torch.Tensor) -> torch.Tensor:
    """Straight-through Gumbel sampling for differentiable token selection."""
    # Sample from Gumbel distribution
    gumbel = -torch.log(-torch.log(torch.rand_like(probs)))
    # Get argmax indices
    indices = (probs + gumbel).argmax(dim=-1)
    # One-hot encoding with straight-through gradient
    one_hot = F.one_hot(indices, num_classes=probs.size(-1)).float()
    # Straight-through: forward pass uses one_hot, backward uses probs
    return one_hot - probs.detach() + probs


class EnergyOracle(nn.Module):
    """Energy-based oracle for differentiable dynamical systems."""

    def __init__(self, vocab_size: int, state_dim: int, tok_embed_dim: int = 768):
        super().__init__()
        self.state_dim = state_dim
        self.tok_embed_dim = tok_embed_dim

        # Token embedding for expected value computation - use model's embeddings
        self.tok_embed = nn.Parameter(torch.zeros(vocab_size, tok_embed_dim))

        # State transition function f(s_t, u_t) -> s_{t+1}
        self.f = nn.Sequential(
            nn.Linear(state_dim + tok_embed_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim)
        )

        # Energy function E(s_t, v) -> scalar per token
        self.E = nn.Sequential(
            nn.Linear(state_dim + tok_embed_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1)  # Single energy value per state-token pair
        )

        # Initial state (learnable)
        self.initial_state = nn.Parameter(torch.randn(state_dim) * 0.02)

    def energies(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for each token given current state.

        Args:
            s_t: Current state [batch_size, state_dim]

        Returns:
            Energy tensor [batch_size, vocab_size]
        """
        batch_size = s_t.size(0)
        vocab_size = self.tok_embed.size(0)

        # For memory efficiency, process in chunks to avoid OOM
        chunk_size = min(4096, vocab_size)  # Adjust based on memory constraints
        energies_list = []

        for start_idx in range(0, vocab_size, chunk_size):
            end_idx = min(start_idx + chunk_size, vocab_size)

            # Get chunk of token embeddings
            tok_chunk = self.tok_embed[start_idx:end_idx]  # [chunk_size, tok_embed_dim]

            # Expand state for this chunk
            s_expanded = s_t.unsqueeze(1).expand(-1, tok_chunk.size(0), -1)  # [B, chunk_size, state_dim]
            tok_expanded = tok_chunk.unsqueeze(0).expand(batch_size, -1, -1)  # [B, chunk_size, tok_embed_dim]

            # Concatenate state and token embedding for this chunk
            state_tok = torch.cat([s_expanded, tok_expanded], dim=-1)  # [B, chunk_size, state_dim + tok_embed_dim]

            # Compute energy for this chunk
            chunk_energies = self.E(state_tok).squeeze(-1)  # [B, chunk_size]
            energies_list.append(chunk_energies)

        # Concatenate all chunks
        energies = torch.cat(energies_list, dim=1)  # [B, V]

        return energies

    def roll(self, s_t: torch.Tensor, p_t: torch.Tensor, mode: str = 'expected') -> torch.Tensor:
        """
        Roll state forward given token probabilities.

        Args:
            s_t: Current state [batch_size, state_dim]
            p_t: Token probabilities [batch_size, vocab_size]
            mode: 'expected' or 'st-gumbel'

        Returns:
            Next state [batch_size, state_dim]
        """
        if mode == 'expected':
            # Expected token embedding
            u_t = p_t @ self.tok_embed  # [batch_size, tok_embed_dim]
        elif mode == 'st-gumbel':
            # Straight-through Gumbel sample
            one_hot = straight_through_sample(p_t)
            u_t = one_hot @ self.tok_embed  # [batch_size, tok_embed_dim]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Concatenate state and token embedding
        state_input = torch.cat([s_t, u_t], dim=-1)  # [batch_size, state_dim + tok_embed_dim]

        # Apply transition function
        s_next = self.f(state_input)  # [batch_size, state_dim]

        return s_next


class InsideOutside:
    """Inside-outside algorithm for PCFG in log space."""

    def __init__(self):
        pass

    def inside_log(self, Theta: torch.Tensor, Phi: torch.Tensor, start_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inside probabilities in log space.

        Args:
            Theta: Rule probabilities [N, N, N] where N is num_nonterminals
            Phi: Terminal probabilities [B, N, T] for batch B, nonterminals N, sequence length T
            start_id: Index of start symbol

        Returns:
            beta: Inside probabilities [B, N, T, T]
            logZ: Log partition function [B]
        """
        # Stub implementation - returns zero tensors
        B, N, T = Phi.shape
        beta = torch.zeros(B, N, T, T, device=Phi.device, dtype=Phi.dtype)
        logZ = torch.zeros(B, device=Phi.device, dtype=Phi.dtype)
        return beta, logZ

    def outside_log(self, Theta: torch.Tensor, Phi: torch.Tensor, beta: torch.Tensor, start_id: int = 0) -> torch.Tensor:
        """
        Compute outside probabilities in log space.

        Args:
            Theta: Rule probabilities [N, N, N]
            Phi: Terminal probabilities [B, N, T]
            beta: Inside probabilities [B, N, T, T]
            start_id: Index of start symbol

        Returns:
            alpha: Outside probabilities [B, N, T, T]
        """
        B, N, T = Phi.shape
        alpha = torch.zeros(B, N, T, T, device=Phi.device, dtype=Phi.dtype)
        return alpha

    def term_marginals(self, alpha: torch.Tensor, beta: torch.Tensor, logZ: torch.Tensor) -> torch.Tensor:
        """
        Compute terminal marginal probabilities.

        Args:
            alpha: Outside probabilities [B, N, T, T]
            beta: Inside probabilities [B, N, T, T]
            logZ: Log partition function [B]

        Returns:
            p_nt: Marginal probabilities for nonterminals [B, N, T]
        """
        B, N, T = alpha.shape[:3]
        p_nt = torch.zeros(B, N, T, device=alpha.device, dtype=alpha.dtype)
        return p_nt


class CFGOracle(nn.Module):
    """CFG-based oracle for generating soft masks from governed sequences."""

    def __init__(self, num_nonterminals: int, vocab_size: int, nt2vocab: torch.Tensor):
        super().__init__()
        self.N = num_nonterminals
        self.V = vocab_size

        # Rule probabilities Theta[r,s,t] = P(r -> s t)
        self.Theta = nn.Parameter(torch.randn(num_nonterminals, num_nonterminals, num_nonterminals) * 0.01)

        # Mapping from nonterminals to vocabulary items (0/1 mask) - as buffer, not parameter
        self.register_buffer('nt2vocab', nt2vocab.float())

        # Terminal head - built later when k is known
        self.term_head = None

        # Inside-outside algorithm
        self.parser = InsideOutside()

    def build_term_head(self, k: int):
        """Build terminal head with correct input dimension."""
        self.term_head = nn.Sequential(
            nn.LayerNorm(k),
            nn.Linear(k, 128),
            nn.GELU(),
            nn.Linear(128, self.N)
        )

    def mask_and_loss(self, g_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CFG-based mask and loss for a governed sequence.

        Args:
            g_seq: Governed sequence [batch_size, seq_len, k]

        Returns:
            mask: Soft mask [batch_size, seq_len, vocab_size]
            L_pcfg: PCFG loss (negative logZ)
        """
        if self.term_head is None:
            raise ValueError("Call build_term_head(k) first to initialize the terminal head")

        B, T, k = g_seq.shape

        # Get terminal probabilities Phi [B, N, T]
        # Apply operations step by step to handle dimension ordering correctly
        B, T, k = g_seq.shape

        # Reshape for LayerNorm: [B*T, k]
        g_flat = g_seq.view(-1, k)  # [B*T, k]
        g_norm = self.term_head[0](g_flat)  # LayerNorm: [B*T, k]

        # Apply linear layers: [B*T, k] -> [B*T, 128]
        g_linear = self.term_head[1](g_norm)  # Linear(k, 128)
        g_activated = self.term_head[2](g_linear)  # GELU
        g_output = self.term_head[3](g_activated)  # Linear(128, N)

        # Reshape back to [B, N, T]
        Phi = g_output.view(B, T, -1).transpose(1, 2)  # [B, T, N] -> [B, N, T]
        Phi = F.log_softmax(Phi, dim=1)  # Log probabilities

        # Convert Theta to probabilities in log space
        Theta_log = F.log_softmax(self.Theta.view(-1, self.N), dim=-1).view(self.N, self.N, self.N)

        # Run inside-outside algorithm
        beta, logZ = self.parser.inside_log(Theta_log, Phi, start_id=0)
        alpha = self.parser.outside_log(Theta_log, Phi, beta, start_id=0)

        # Get marginal probabilities for terminals (length-1 spans)
        p_nt = self.parser.term_marginals(alpha, beta, logZ)  # [B, N, T]

        # Convert to vocabulary mask: log P(vocab v | nonterm nt at position t)
        # mask = logsumexp over nonterminals of p_nt * nt2vocab
        mask_logits = torch.log(1e-6 + torch.einsum('bnt,nv->btv', p_nt, self.nt2vocab))

        # PCFG loss (negative log likelihood)
        L_pcfg = (-logZ).mean()

        return mask_logits, L_pcfg


class ProgramOracle:
    """Program oracle for arbitrary hard constraints on token sequences."""

    def __init__(self, tokenizer=None, constraint_fn=None):
        """
        Initialize program oracle.

        Args:
            tokenizer: Tokenizer for converting tokens to strings if needed
            constraint_fn: Function that takes prefix strings and returns valid token masks
                         Should return tensor [batch_size, seq_len, vocab_size] of {0, -inf}
        """
        self.tokenizer = tokenizer
        self.constraint_fn = constraint_fn

    def valid_mask(self, prefixes: List[str], device=None) -> torch.Tensor:
        """
        Compute valid token mask for given prefixes.

        Args:
            prefixes: List of prefix strings [batch_size]
            device: Device for the returned tensor

        Returns:
            mask: Valid token mask [batch_size, vocab_size] where valid tokens have 0, invalid have -inf
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.constraint_fn is None:
            # No constraints - allow all tokens
            batch_size = len(prefixes)
            vocab_size = self.tokenizer.vocab_size if self.tokenizer else 50000
            return torch.zeros(batch_size, vocab_size, device=device)

        # Apply custom constraint function
        return self.constraint_fn(prefixes)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze()  # For causal LM
        }


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)

    def get_metrics(self) -> Dict[str, List]:
        """Get all tracked metrics."""
        return dict(self.metrics)

    def get_latest(self, key: str):
        """Get latest value for a metric."""
        values = self.metrics.get(key, [])
        return values[-1] if values else None

    def compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for all metrics."""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1]
                }
        return stats

    def save_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'metrics': self.get_metrics(),
            'stats': self.compute_stats(),
            'elapsed_time': time.time() - self.start_time
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class OPTWithGovernance(nn.Module):
    """OPT model wrapper with subspace governance and oracle integration."""

    def __init__(self, model, subgov: SubspaceGovernance, energy_oracle: EnergyOracle,
                 cfg_oracle: CFGOracle, program_oracle: ProgramOracle, config: TrainingConfig,
                 state_encoder: nn.Module):
        super().__init__()
        self.model = model
        self.subgov = subgov
        self.energy_oracle = energy_oracle
        self.cfg_oracle = cfg_oracle
        self.program_oracle = program_oracle
        self.config = config
        self.state_encoder = state_encoder

        # Storage for captured hidden states
        self.captured_layers = []

        # Initialize logit fusion heads
        self._init_logit_heads()

        # Register hooks for capturing hidden states
        self._register_hooks()

    def _init_logit_heads(self):
        """Initialize logit heads for governed and free subspaces at final port."""
        # Get final port dimensions
        final_k = self.subgov.k_schedule[-1]  # Last entry is final port k

        # Governed subspace head: maps g_final to logits
        self.Wg = nn.Linear(final_k, self.model.lm_head.out_features, bias=False)

        # Free subspace head: maps f_final to logits (initialize from original lm_head)
        self.Wf = nn.Linear(self.model.config.hidden_size, self.model.lm_head.out_features, bias=False)

        # Copy original lm_head weights to Wf
        with torch.no_grad():
            self.Wf.weight.copy_(self.model.lm_head.weight)

    def _register_hooks(self):
        """Register forward hooks to capture hidden states at governed layers."""
        # Access the decoder layers
        decoder = self.model.model.decoder

        # Determine which layers to govern
        if self.config.govern.layers is None:
            # Govern all layers + final
            govern_layers = list(range(len(decoder.layers))) + ['final']
        else:
            govern_layers = self.config.govern.layers

        # Create hooks for specified layers
        for layer_idx in govern_layers:
            if layer_idx == 'final':
                # Skip final hook - we'll use outputs.hidden_states[-1] instead
                pass
            else:
                # Hook for intermediate layers
                def hook(module, inp, out, idx=layer_idx):
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    self.captured_layers.append((h, idx))
                decoder.layers[layer_idx].register_forward_hook(hook)

    def forward(self, input_ids, attention_mask=None, labels=None, return_governance_info=False, scales=None):
        """Forward pass with subspace governance and oracle fusion."""
        # Clear previous captures
        self.captured_layers.clear()

        # Enable hidden states output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        # Get final hidden state (after last decoder layer, before lm_head)
        final_hidden_state = outputs.hidden_states[-1] if outputs.hidden_states else None

        # Apply subspace governance and logit fusion
        loss, logits, aux_info = self._apply_governance_and_fusion(outputs, labels, scales, final_hidden_state)

        if return_governance_info:
            return type('Output', (), {'loss': loss, 'logits': logits, 'aux_info': aux_info})()
        else:
            return type('Output', (), {'loss': loss, 'logits': logits})()

    def _apply_governance_and_fusion(self, outputs, labels, scales=None, final_hidden_state=None):
        """Apply subspace governance and oracle-based logit fusion."""
        batch_size, seq_len = labels.shape[:2] if labels is not None else outputs.logits.shape[:2]
        scales = scales or {}
        step = int(scales.get('step', 0))

        # Project final hidden state into governed and free subspaces
        gF, fF, Ufinal, gkF = self.subgov.project(final_hidden_state, len(self.subgov.k_schedule) - 1)

        # Base logits from both subspaces
        logits_base = self.Wg(gkF) + self.Wf(fF)  # [B, T, V]

        # Initialize auxiliary info
        aux_info = {
            'logZ': 0.0,
            'energy_term': 0.0,
            'cfg_term': 0.0,
            'regularizer_terms': {},
            'eig_stats': {}
        }

        # Initialize contributions as device-safe tensors
        device = logits_base.device
        m_energy = torch.zeros_like(logits_base)
        m_cfg = torch.zeros_like(logits_base)
        m_prog = torch.zeros_like(logits_base)
        L_energy = torch.tensor(0.0, device=device)
        L_pcfg = torch.tensor(0.0, device=device)
        lambda_cfg = 0.0

        # Energy oracle contribution
        if self.config.energy.enabled:
            # Get beta from scales (computed in trainer)
            beta = float(scales.get('beta', self.config.energy.beta))

            # Use dynamic state from governed coordinates at final port (per-position)
            s_seq = self.state_encoder(gkF)  # [B, T, state_dim]
            # For now, use the last timestep state (can be extended to per-position later)
            s_t = s_seq[:, -1, :]  # [B, state_dim]
            E = self.energy_oracle.energies(s_t)  # [B, V]
            E_expanded = E.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, V]
            m_energy = -beta * E_expanded

            if self.training and self.config.energy.mode == 'one-step':
                # 1-step energy loss - use position-aware energies
                gold = labels[:, 1:]  # [B, T-1]
                goldE = E_expanded[:, :-1, :].gather(-1, gold.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
                L_energy = beta * goldE.mean()
                aux_info['energy_term'] = L_energy.item()
                aux_info['beta'] = beta

        # CFG oracle contribution
        if self.config.cfg.enabled:
            # Get lambda from scales (computed in trainer)
            lambda_cfg = float(scales.get('lambda_cfg', self.config.cfg.lambda_))

            # Use governed sequence from a middle layer for CFG
            if self.captured_layers:
                # Use the first captured layer for CFG input
                h_mid, mid_idx = self.captured_layers[0]
                _, _, _, gk_mid = self.subgov.project(h_mid, mid_idx)
                m_cfg, L_pcfg = self.cfg_oracle.mask_and_loss(gk_mid)
                aux_info['logZ'] = -L_pcfg.item()  # Negative for consistency
                aux_info['cfg_term'] = L_pcfg.item()
                aux_info['lambda_cfg'] = lambda_cfg

        # Program oracle contribution (hard mask)
        if self.config.program.enabled:
            # Convert token ids to strings for program oracle
            # For teacher forcing, use the full sequence up to each position
            prefix_strings = []
            for b in range(batch_size):
                # Decode the entire sequence for this batch item
                token_ids = labels[b, :seq_len].tolist()
                # Remove padding tokens if present
                tok = self.program_oracle.tokenizer
                if tok is not None and tok.pad_token_id is not None:
                    token_ids = [tid for tid in token_ids if tid != tok.pad_token_id]
                try:
                    if tok is not None:
                        text = tok.decode(token_ids, skip_special_tokens=True)
                    else:
                        text = ""
                    prefix_strings.append(text)
                except:
                    # Fallback for decoding errors
                    prefix_strings.append("")

            m_prog = self.program_oracle.valid_mask(prefix_strings, device=logits_base.device)
            if m_prog.dim() == 2:  # [B, V] -> [B, 1, V]
                m_prog = m_prog.unsqueeze(1)
            m_prog = m_prog.expand(-1, seq_len, -1)  # [B, T, V]

        # Combine all logit contributions
        logits = logits_base + m_energy + lambda_cfg * m_cfg + m_prog

        # Compute main task loss
        if labels is not None:
            L_task = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
        else:
            L_task = 0

        # Add oracle losses
        total_loss = L_task + L_energy + lambda_cfg * L_pcfg

        # Add regularizers
        reg_loss = self._compute_regularizers(final_hidden_state)
        orth_loss = self.config.regularizers.orth_weight * self.subgov.orth_penalty()
        total_loss += reg_loss + orth_loss
        aux_info['regularizer_terms'] = {
            'anti_collapse': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'orthogonality': orth_loss.item()
        }

        # Compute eigenvalue statistics for governed coordinates (gated for performance)
        # Only compute every 100 steps to avoid expensive eigendecomposition on every batch
        if step % 100 == 0:
            with torch.no_grad():
                if gkF.numel() > 0:
                    x = gkF.reshape(-1, gkF.size(-1)) - gkF.reshape(-1, gkF.size(-1)).mean(0, keepdim=True)
                    cov = (x.T @ x) / max(x.size(0) - 1, 1)
                    k = cov.shape[0]
                    evals = torch.linalg.eigvalsh(cov)
                    trace_cov = torch.trace(cov)
                    avg_variance = trace_cov / k
                    utilization_threshold = 1e-2  # Small threshold for "active" dimensions
                    utilization_ratio = (evals > utilization_threshold).float().mean()

                    aux_info['eig_stats'] = {
                        'min': float(evals.min()),
                        'median': float(evals.median()),
                        'max': float(evals.max()),
                        'avg_variance': float(avg_variance),
                        'utilization_ratio': float(utilization_ratio)
                    }
                else:
                    aux_info['eig_stats'] = {}
        else:
            aux_info['eig_stats'] = {}

        return total_loss, logits, aux_info

    def _compute_regularizers(self, final_hidden_state):
        """Compute anti-collapse and orthogonality regularizers."""
        reg_loss = 0.0

        # Anti-collapse regularizers for all governed layers
        for h_l, layer_idx in self.captured_layers:
            g_l, f_l, U_l, gk_l = self.subgov.project(h_l, layer_idx)

            # Variance floor
            var_loss = self._var_floor(gk_l, tau=self.config.regularizers.var_tau)
            reg_loss += self.config.regularizers.var_weight * var_loss

            # Logdet spread
            logdet_loss = self._logdet_spread(gk_l, delta=self.config.regularizers.logdet_delta)
            reg_loss += self.config.regularizers.logdet_weight * logdet_loss

            # Radial floor
            radial_loss = self._radial_floor(gk_l, rho=self.config.regularizers.radial_rho)
            reg_loss += self.config.regularizers.radial_weight * radial_loss

        # Final layer regularizers
        if final_hidden_state is not None:
            gF, fF, Ufinal, gkF = self.subgov.project(final_hidden_state,
                                                    len(self.subgov.k_schedule) - 1)

            var_loss = self._var_floor(gkF, tau=self.config.regularizers.var_tau)
            reg_loss += self.config.regularizers.var_weight * var_loss

            logdet_loss = self._logdet_spread(gkF, delta=self.config.regularizers.logdet_delta)
            reg_loss += self.config.regularizers.logdet_weight * logdet_loss

            radial_loss = self._radial_floor(gkF, rho=self.config.regularizers.radial_rho)
            reg_loss += self.config.regularizers.radial_weight * radial_loss

        return reg_loss

    def _var_floor(self, g: torch.Tensor, tau: float = 1.0, eps: float = 1e-4):
        """Variance floor regularizer."""
        x = g.reshape(-1, g.shape[-1])
        x = x - x.mean(0, keepdim=True)
        std = (x.pow(2).mean(0) + eps).sqrt()
        return F.relu(tau - std).mean()

    def _logdet_spread(self, g: torch.Tensor, delta: float = 1e-3):
        """Log-determinant spread regularizer."""
        x = g.reshape(-1, g.shape[-1])
        cov = (x.T @ x) / max(x.shape[0] - 1, 1)
        k = cov.shape[0]
        cov = cov + delta * torch.eye(k, device=g.device, dtype=g.dtype)
        sign, logabsdet = torch.slogdet(cov)
        return -logabsdet

    def _radial_floor(self, g: torch.Tensor, rho: float = 0.25):
        """Radial floor regularizer."""
        return F.relu(rho - g.norm(dim=-1)).mean()

    def generate(self, *args, **kwargs):
        """Generation method (for eval - bypasses governance during generation)."""
        return self.model.generate(*args, **kwargs)


class OPTFinetunerWithGovernance:
    """Main class for finetuning OPT with subspace governance."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set random seeds and deterministic settings
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # Enable deterministic CUDA for reproducibility (if CUDA available)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set padding side for generation with attention masks
        self.tokenizer.padding_side = "left"

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.float32
        ).to(self.device)

        # Disable cache during training (saves memory, avoids HF warnings)
        self.model.config.use_cache = False

        # Freeze embeddings for parameter efficiency
        for param in self.model.model.decoder.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model.model.decoder.embed_positions.parameters():
            param.requires_grad = False

        # Initialize subspace governance
        self.subgov = SubspaceGovernance(config)

        # Initialize oracles
        vocab_size = self.model.lm_head.out_features
        self.energy_oracle = EnergyOracle(
            vocab_size=vocab_size,
            state_dim=config.energy.state_dim,
            tok_embed_dim=self.model.config.hidden_size
        )

        # Initialize token embeddings with model's lm_head weights
        with torch.no_grad():
            self.energy_oracle.tok_embed.copy_(self.model.lm_head.weight)

        # Initialize state encoder for dynamic energy state
        final_k = self.subgov.k_schedule[-1]  # Final port k dimension
        self.state_encoder = nn.Sequential(
            nn.Linear(final_k, self.config.energy.state_dim),
            nn.Tanh()
        ).to(self.device)

        # Create CFG oracle
        if config.cfg.nt2vocab_source == "all_tokens":
            # Allow all tokens for all nonterminals initially
            nt2vocab = torch.ones(config.cfg.N, vocab_size)
        else:
            # Would need custom logic for specific mappings
            nt2vocab = torch.ones(config.cfg.N, vocab_size)

        self.cfg_oracle = CFGOracle(
            num_nonterminals=config.cfg.N,
            vocab_size=vocab_size,
            nt2vocab=nt2vocab
        )
        # Build terminal head with correct k dimension (use first layer's k for CFG)
        first_k = self.subgov.k_schedule[0] if len(self.subgov.k_schedule) > 0 else 64
        self.cfg_oracle.build_term_head(first_k)

        # Initialize program oracle
        self.program_oracle = ProgramOracle(tokenizer=self.tokenizer)

        # Wrap model with governance
        self.model_with_governance = OPTWithGovernance(
            self.model, self.subgov, self.energy_oracle,
            self.cfg_oracle, self.program_oracle, config, self.state_encoder
        ).to(self.device)

        # Setup optimizers
        self.setup_optimizers()

        # Setup metrics
        self.metrics = MetricsTracker()

        # Training state
        self.global_step = 0

    def setup_optimizers(self):
        """Setup optimizers for governance training."""
        # Parameters to optimize: base model, subspace governance, energy oracle, CFG oracle, fusion heads
        trainable_params = []
        trainable_params += [p for p in self.model.parameters() if p.requires_grad]
        trainable_params += list(self.subgov.parameters())
        trainable_params += list(self.energy_oracle.parameters())
        trainable_params += list(self.cfg_oracle.parameters())
        trainable_params += list(self.state_encoder.parameters())
        trainable_params += list(self.model_with_governance.Wg.parameters())
        trainable_params += list(self.model_with_governance.Wf.parameters())

        if self.config.optimizer_type.lower() == "muon":
            self.optimizer = MuonOptimizer(
                trainable_params,
                lr=self.config.learning_rate * self.config.muon_lr_scale,
                weight_decay=self.config.weight_decay
            )
            print(f"Using Muon optimizer with lr={self.config.learning_rate * self.config.muon_lr_scale}")
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
            print(f"Using AdamW optimizer with lr={self.config.learning_rate}")

        # Learning rate scheduler
        num_training_steps = self.config.num_epochs * (self.config.dataset_size // self.config.batch_size)
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )

    def load_sample_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Load a dataset for finetuning from HuggingFace Hub."""
        print(f"Loading dataset with {self.config.dataset_size} examples...")

        # Load a real dataset from HuggingFace
        # We'll use wikitext as it's clean and appropriate for language modeling
        print("Loading WikiText dataset from HuggingFace Hub...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        # Filter out empty lines and very short texts
        def filter_text(example):
            return len(example['text'].strip()) > 50  # At least 50 characters

        dataset = dataset.filter(filter_text)

        # Sample a subset if we want a specific size
        if self.config.dataset_size < len(dataset):
            # Sample random subset
            indices = random.sample(range(len(dataset)), self.config.dataset_size)
            dataset = dataset.select(indices)
            print(f"Sampled {self.config.dataset_size} examples from {len(dataset)} available")

        # Convert to list of texts
        texts = dataset['text']

        print(f"Loaded {len(texts)} text examples from WikiText dataset")
        print(f"Sample text: {texts[0][:100]}...")

        # Create dataset
        dataset = TextDataset(texts, self.tokenizer, self.config.max_length)

        # Split into train/val
        train_size = int(self.config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )

        print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val examples")
        return train_loader, val_loader

    def compute_loss(self, batch: Dict[str, torch.Tensor], return_aux_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Compute loss for a batch with governance."""
        # Compute warmup scales
        warm = max(1, self.config.warmup_steps)
        beta = self.config.energy.beta * min(1.0, self.global_step / warm)
        lambda_cfg = self.config.cfg.lambda_ * min(1.0, self.global_step / warm)

        output = self.model_with_governance(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            return_governance_info=return_aux_info,
            scales={'beta': beta, 'lambda_cfg': lambda_cfg, 'step': self.global_step}
        )

        if return_aux_info:
            return output.loss, output.aux_info
        else:
            return output.loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step with governance."""
        self.model.train()
        self.model_with_governance.train()
        self.subgov.train()
        self.energy_oracle.train()
        self.cfg_oracle.train()

        self.optimizer.zero_grad()

        # Compute warmup scales
        warm = max(1, self.config.warmup_steps)
        beta = self.config.energy.beta * min(1.0, self.global_step / warm)
        lambda_cfg = self.config.cfg.lambda_ * min(1.0, self.global_step / warm)

        # Forward pass with scales
        output = self.model_with_governance(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            return_governance_info=True,
            scales={'beta': beta, 'lambda_cfg': lambda_cfg, 'step': self.global_step}
        )
        loss, aux_info = output.loss, output.aux_info

        # Backward pass
        loss.backward()

        # Gradient clipping - capture norm for logging
        total_norm = torch.nn.utils.clip_grad_norm_(self.model_with_governance.parameters(), self.config.gradient_clip_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Update metrics
        metrics = {
            'train_loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'grad_norm': float(total_norm),
            'logZ': aux_info.get('logZ', 0.0),
            'energy_term': aux_info.get('energy_term', 0.0),
            'cfg_term': aux_info.get('cfg_term', 0.0),
            'beta': aux_info.get('beta', 0.0),
            'lambda_cfg': aux_info.get('lambda_cfg', 0.0)
        }

        return metrics

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one evaluation step."""
        self.model.eval()
        self.model_with_governance.eval()
        self.subgov.eval()
        self.energy_oracle.eval()
        self.cfg_oracle.eval()

        with torch.no_grad():
            output = self.model_with_governance(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                return_governance_info=True
            )
            loss, aux_info = output.loss, output.aux_info

        return {
            'val_loss': loss.item(),
            'val_logZ': aux_info.get('logZ', 0.0),
            'val_energy_term': aux_info.get('energy_term', 0.0),
            'val_cfg_term': aux_info.get('cfg_term', 0.0),
            'val_beta': aux_info.get('beta', 0.0),
            'val_lambda_cfg': aux_info.get('lambda_cfg', 0.0)
        }

    def generate_sample(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate sample text for qualitative evaluation."""
        self.model.eval()
        # Use base model for generation (bypass governance for speed/quality)
        self.model_with_governance.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.model_with_governance.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

    def train(self):
        """Main training loop with subspace governance."""
        print("=" * 60)
        print(f"STARTING FINETUNING: {self.config.optimizer_type.upper()} WITH SUBSPACE GOVERNANCE")
        print("=" * 60)

        # Display governance configuration
        print(f"Subspace governance k-schedule: {self.subgov.get_k_schedule()}")
        print(f"Energy enabled: {self.config.energy.enabled}")
        print(f"CFG enabled: {self.config.cfg.enabled}")
        print(f"Program enabled: {self.config.program.enabled}")

        # Load data
        train_loader, val_loader = self.load_sample_dataset()

        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)

            # Training
            self.model.train()
            self.model_with_governance.train()
            self.subgov.train()
            self.energy_oracle.train()
            self.cfg_oracle.train()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                metrics = self.train_step(batch)
                self.metrics.update(**metrics)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = np.mean(self.metrics.metrics['train_loss'][-self.config.log_steps:])
                    print(f"Step {self.global_step:5d} | Loss: {avg_loss:.4f} | LR: {metrics['learning_rate']:.6f}")
                    if 'logZ' in metrics and metrics['logZ'] != 0:
                        print(f"                      | logZ: {metrics['logZ']:.2f} | Energy: {metrics.get('energy_term', 0):.4f}")

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate(val_loader)
                    self.metrics.update(**val_metrics)

                    print(f"Step {self.global_step:5d} | Val Loss: {val_metrics['val_loss']:.4f}")
                    if 'val_logZ' in val_metrics:
                        print(f"                      | Val logZ: {val_metrics['val_logZ']:.2f}")

                    # Generate sample
                    sample_text = self.generate_sample("The future of AI")
                    print(f"Sample generation: {sample_text[:100]}...")

            # End of epoch evaluation
            val_metrics = self.evaluate(val_loader)
            self.metrics.update(**val_metrics)

            print(f"Epoch {epoch + 1} complete | Val Loss: {val_metrics['val_loss']:.4f}")

        # Final evaluation
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        final_val_metrics = self.evaluate(val_loader)
        print(f"Final validation loss: {final_val_metrics['val_loss']:.4f}")

        # Generate final samples
        print("\nFinal sample generations:")
        for prompt in ["The future of artificial intelligence", "Machine learning is", "Neural networks can"]:
            sample = self.generate_sample(prompt, max_new_tokens=30)
            print(f"Prompt: {prompt}")
            print(f"Generated: {sample}\n")

        # Save results
        self.save_results()

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set with governance."""
        self.model.eval()
        self.model_with_governance.eval()
        self.subgov.eval()
        self.energy_oracle.eval()
        self.cfg_oracle.eval()

        total_loss = 0
        total_logZ = 0
        total_energy = 0
        total_cfg = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, aux_info = self.compute_loss(batch, return_aux_info=True)
                total_loss += loss.item()
                total_logZ += aux_info.get('logZ', 0.0)
                total_energy += aux_info.get('energy_term', 0.0)
                total_cfg += aux_info.get('cfg_term', 0.0)
                num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_logZ': total_logZ / num_batches,
            'val_energy_term': total_energy / num_batches,
            'val_cfg_term': total_cfg / num_batches
        }

    def save_results(self):
        """Save training results and metrics."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"{self.config.run_name}_metrics.json")
        self.metrics.save_to_file(metrics_file)

        # Save config
        config_file = os.path.join(self.config.output_dir, f"{self.config.run_name}_config.json")
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save model
        model_file = os.path.join(self.config.output_dir, f"{self.config.run_name}_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'subgov_state_dict': self.subgov.state_dict(),
            'energy_oracle_state_dict': self.energy_oracle.state_dict(),
            'cfg_oracle_state_dict': self.cfg_oracle.state_dict(),
            'governance_state_dict': self.model_with_governance.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'global_step': self.global_step
        }, model_file)

        # Save subspace governance module separately for analysis
        subgov_file = os.path.join(self.config.output_dir, f"{self.config.run_name}_subspace_governance.pt")
        torch.save({
            'subgov_state_dict': self.subgov.state_dict(),
            'k_schedule': self.subgov.get_k_schedule(),
            'projection_matrices': [proj.weight.data.clone() for proj in self.subgov.proj_matrices]
        }, subgov_file)

        print(f"Results saved to {self.config.output_dir}")
        print(f"Metrics: {metrics_file}")
        print(f"Model: {model_file}")
        print(f"Subspace governance: {subgov_file}")

    def plot_training_curves(self):
        """Plot training metrics."""
        metrics = self.metrics.get_metrics()

        # Create subplots based on available metrics
        num_plots = sum(1 for key in ['train_loss', 'val_loss', 'logZ', 'energy_term', 'cfg_term', 'beta', 'lambda_cfg', 'learning_rate', 'grad_norm']
                       if key in metrics)

        if num_plots == 0:
            return

        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        plot_idx = 0

        # Training loss
        if 'train_loss' in metrics:
            axes[plot_idx].plot(metrics['train_loss'])
            axes[plot_idx].set_title('Training Loss')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Loss')
            plot_idx += 1

        # Validation loss
        if 'val_loss' in metrics:
            val_steps = list(range(0, len(metrics['val_loss']) * self.config.eval_steps, self.config.eval_steps))[:len(metrics['val_loss'])]
            axes[plot_idx].plot(val_steps, metrics['val_loss'])
            axes[plot_idx].set_title('Validation Loss')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Loss')
            plot_idx += 1

        # logZ (CFG partition function)
        if 'logZ' in metrics and any(x != 0 for x in metrics['logZ']):
            logZ_steps = list(range(0, len(metrics['logZ']) * self.config.log_steps, self.config.log_steps))[:len(metrics['logZ'])]
            axes[plot_idx].plot(logZ_steps, metrics['logZ'])
            axes[plot_idx].set_title('CFG logZ (Train)')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('logZ')
            plot_idx += 1

        # Energy term
        if 'energy_term' in metrics and any(x != 0 for x in metrics['energy_term']):
            energy_steps = list(range(0, len(metrics['energy_term']) * self.config.log_steps, self.config.log_steps))[:len(metrics['energy_term'])]
            axes[plot_idx].plot(energy_steps, metrics['energy_term'])
            axes[plot_idx].set_title('Energy Term')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Energy')
            plot_idx += 1

        # CFG term
        if 'cfg_term' in metrics and any(x != 0 for x in metrics['cfg_term']):
            cfg_steps = list(range(0, len(metrics['cfg_term']) * self.config.log_steps, self.config.log_steps))[:len(metrics['cfg_term'])]
            axes[plot_idx].plot(cfg_steps, metrics['cfg_term'])
            axes[plot_idx].set_title('CFG Term')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('CFG Loss')
            plot_idx += 1

        # Beta schedule
        if 'beta' in metrics and any(x != 0 for x in metrics['beta']):
            beta_steps = list(range(0, len(metrics['beta']) * self.config.log_steps, self.config.log_steps))[:len(metrics['beta'])]
            axes[plot_idx].plot(beta_steps, metrics['beta'])
            axes[plot_idx].set_title('Energy Beta')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Beta')
            plot_idx += 1

        # Lambda CFG schedule
        if 'lambda_cfg' in metrics and any(x != 0 for x in metrics['lambda_cfg']):
            lambda_steps = list(range(0, len(metrics['lambda_cfg']) * self.config.log_steps, self.config.log_steps))[:len(metrics['lambda_cfg'])]
            axes[plot_idx].plot(lambda_steps, metrics['lambda_cfg'])
            axes[plot_idx].set_title('CFG Lambda')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Lambda')
            plot_idx += 1

        # Learning rate
        if 'learning_rate' in metrics:
            axes[plot_idx].plot(metrics['learning_rate'])
            axes[plot_idx].set_title('Learning Rate')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('LR')
            plot_idx += 1

        # Gradient norm
        if 'grad_norm' in metrics:
            axes[plot_idx].plot(metrics['grad_norm'])
            axes[plot_idx].set_title('Gradient Norm')
            axes[plot_idx].set_xlabel('Step')
            axes[plot_idx].set_ylabel('Norm')
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.run_name}_training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory


def main():
    """Main function to run finetuning experiments."""
    parser = argparse.ArgumentParser(description="Finetune OPT-125M with Subspace Governance")
    parser.add_argument("--optimizer", type=str, choices=["muon", "adam"], default="muon",
                       help="Optimizer to use (default: muon)")
    parser.add_argument("--dataset_size", type=int, default=1000,
                       help="Size of dataset to use (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs (default: 3)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--run_name", type=str, default="opt_muon_subspace_governance_experiment",
                       help="Name for this run (default: opt_muon_subspace_governance_experiment)")

    # Subspace governance options
    parser.add_argument("--energy_enabled", action="store_true",
                       help="Enable energy oracle")
    parser.add_argument("--cfg_enabled", action="store_true",
                       help="Enable CFG oracle")
    parser.add_argument("--program_enabled", action="store_true",
                       help="Enable program oracle")
    parser.add_argument("--k_max", type=int, default=128,
                       help="Maximum k dimension for subspace governance (default: 128)")
    parser.add_argument("--energy_beta", type=float, default=1.0,
                       help="Energy beta parameter (default: 1.0)")
    parser.add_argument("--cfg_lambda", type=float, default=0.1,
                       help="CFG lambda parameter (default: 0.1)")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        optimizer_type=args.optimizer,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
        energy=EnergyConfig(enabled=args.energy_enabled, beta=args.energy_beta),
        cfg=CFGConfig(enabled=args.cfg_enabled, lambda_=args.cfg_lambda),
        program=ProgramConfig(enabled=args.program_enabled),
        govern=GovernConfig(k_max=args.k_max)
    )

    # Run finetuning
    finetuner = OPTFinetunerWithGovernance(config)
    finetuner.train()
    finetuner.plot_training_curves()

    print(f"\nFinetuning complete! Results saved in {config.output_dir}")


# Colab/interactive execution - run with default config if no args provided
if __name__ == "__main__":
    # Check if running in interactive mode (like Colab) or as a script
    import sys

    # Colab passes -f /path/to/kernel.json, so we need to check for this pattern
    is_colab = (
        len(sys.argv) == 1 or
        (len(sys.argv) >= 2 and sys.argv[1].startswith('-f')) or
        (len(sys.argv) == 2 and sys.argv[1] == '--f') or
        any(arg.startswith('-f') for arg in sys.argv)
    )

    if is_colab:
        # Running interactively - use default config
        print("Running in interactive mode with default configuration...")
        print("For custom configuration, use command line arguments.")

        # Create default config for Colab demo
        config = TrainingConfig(
            optimizer_type="muon",
            dataset_size=1000,  # Smaller for Colab demo
            batch_size=4,       # Smaller for Colab demo
            num_epochs=1,       # Just 1 epoch for demo
            learning_rate=1e-4,
            run_name="colab_demo_subspace_governance",
            energy=EnergyConfig(enabled=True, beta=1.0),
            cfg=CFGConfig(enabled=True, lambda_=0.1),
            program=ProgramConfig(enabled=False),  # Disable for demo
            govern=GovernConfig(k_max=64)  # Smaller for demo
        )

        # Run finetuning
        finetuner = OPTFinetunerWithGovernance(config)
        finetuner.train()
        finetuner.plot_training_curves()

        print(f"\nDemo complete! Results saved in {config.output_dir}")
    else:
        # Running as script with arguments
        main()
