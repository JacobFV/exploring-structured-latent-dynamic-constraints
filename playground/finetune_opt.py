#!/usr/bin/env python3
"""
OPT-125M Finetuning with Muon Optimizer
=======================================

This script finetunes OPT-125M on a small dataset using the Muon optimizer
from the paper "Muon Outperforms Adam in Tail-End Associative Memory Learning".

The script includes:
- Muon optimizer implementation based on the paper
- Comprehensive training metrics monitoring
- Comparison with Adam optimizer
- Small dataset handling for efficient experimentation

Usage:
    python playground/finetune_opt.py --optimizer muon --dataset_size 10000
    python playground/finetune_opt.py --optimizer adam --dataset_size 5000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse
import time
import json
import os
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
import random
warnings.filterwarnings("ignore")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not available. Using fallback sample data.")


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
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

    # Logging config
    log_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    output_dir: str = "./opt_finetune_results"

    # Experiment config
    seed: int = 42
    run_name: str = "opt_muon_finetune"


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


class OPTFinetuner:
    """Main class for finetuning OPT with monitoring."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        # Freeze embeddings for parameter efficiency
        for param in self.model.model.decoder.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model.model.decoder.embed_positions.parameters():
            param.requires_grad = False

        # Setup optimizers
        self.setup_optimizers()

        # Setup metrics
        self.metrics = MetricsTracker()

        # Training state
        self.global_step = 0

    def setup_optimizers(self):
        """Setup optimizers for comparison."""
        # Parameters to optimize (excluding frozen embeddings)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

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

        if not HAS_DATASETS:
            raise ImportError("datasets library is required but not available. Please install it with: pip install datasets")

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

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        return outputs.loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        loss = self.compute_loss(batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Update metrics
        metrics = {
            'train_loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'grad_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
        }

        return metrics

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one evaluation step."""
        self.model.eval()
        with torch.no_grad():
            loss = self.compute_loss(batch)

        return {'val_loss': loss.item()}

    def generate_sample(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate sample text for qualitative evaluation."""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
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
        """Main training loop."""
        print("=" * 60)
        print(f"STARTING FINETUNING: {self.config.optimizer_type.upper()} OPTIMIZER")
        print("=" * 60)

        # Load data
        train_loader, val_loader = self.load_sample_dataset()

        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)

            # Training
            self.model.train()
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                metrics = self.train_step(batch)
                self.metrics.update(**metrics)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = np.mean(self.metrics.metrics['train_loss'][-self.config.log_steps:])
                    print(f"Step {self.global_step:5d} | Loss: {avg_loss:.4f} | LR: {metrics['learning_rate']:.6f}")

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate(val_loader)
                    self.metrics.update(**val_metrics)

                    print(f"Step {self.global_step:5d} | Val Loss: {val_metrics['val_loss']:.4f}")

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
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return {'val_loss': total_loss / num_batches}

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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'global_step': self.global_step
        }, model_file)

        print(f"Results saved to {self.config.output_dir}")
        print(f"Metrics: {metrics_file}")
        print(f"Model: {model_file}")

    def plot_training_curves(self):
        """Plot training metrics."""
        metrics = self.metrics.get_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Training loss
        if 'train_loss' in metrics:
            axes[0, 0].plot(metrics['train_loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')

        # Validation loss
        if 'val_loss' in metrics:
            val_steps = [i for i, x in enumerate(metrics.get('train_loss', [])) if (i+1) % (self.config.eval_steps // self.config.log_steps) == 0][:len(metrics['val_loss'])]
            axes[0, 1].plot(metrics['val_loss'])
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Loss')

        # Learning rate
        if 'learning_rate' in metrics:
            axes[1, 0].plot(metrics['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('LR')

        # Gradient norm
        if 'grad_norm' in metrics:
            axes[1, 1].plot(metrics['grad_norm'])
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Norm')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.run_name}_training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory


def main():
    """Main function to run finetuning experiments."""
    parser = argparse.ArgumentParser(description="Finetune OPT-125M with Muon or Adam")
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
    parser.add_argument("--run_name", type=str, default="opt_muon_experiment",
                       help="Name for this run (default: opt_muon_experiment)")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        optimizer_type=args.optimizer,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        run_name=args.run_name
    )

    # Run finetuning
    finetuner = OPTFinetuner(config)
    finetuner.train()
    finetuner.plot_training_curves()

    print(f"\nFinetuning complete! Results saved in {config.output_dir}")


if __name__ == "__main__":
    main()
