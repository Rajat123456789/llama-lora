"""
Real-time memory tracking callback and utilities for training.

Tracks GPU/MPS memory, system RAM, and logs metrics during training.
"""

import torch
import psutil
import os
import time
from typing import Dict, List, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import json


class MemoryTracker:
    """Track memory usage across training."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.history: List[Dict] = []
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved": torch.cuda.memory_reserved() / (1024 ** 2),
                "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2),
            }
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have detailed memory APIs yet
            return {
                "allocated": 0.0,  # Not available
                "reserved": 0.0,   # Not available
                "max_allocated": 0.0,  # Not available
            }
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system RAM usage in MB."""
        mem_info = self.process.memory_info()
        return {
            "rss": mem_info.rss / (1024 ** 2),  # Resident Set Size
            "vms": mem_info.vms / (1024 ** 2),  # Virtual Memory Size
            "percent": self.process.memory_percent(),
        }
    
    def snapshot(self, step: int = 0, loss: Optional[float] = None) -> Dict:
        """Take a memory snapshot."""
        snapshot = {
            "step": step,
            "timestamp": time.time(),
            "gpu": self.get_gpu_memory(),
            "system": self.get_system_memory(),
        }
        if loss is not None:
            snapshot["loss"] = loss
        
        self.history.append(snapshot)
        return snapshot
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.history:
            return {}
        
        gpu_allocated = [h["gpu"]["allocated"] for h in self.history]
        system_rss = [h["system"]["rss"] for h in self.history]
        
        return {
            "gpu_memory_mb": {
                "min": min(gpu_allocated) if gpu_allocated else 0,
                "max": max(gpu_allocated) if gpu_allocated else 0,
                "avg": sum(gpu_allocated) / len(gpu_allocated) if gpu_allocated else 0,
            },
            "system_memory_mb": {
                "min": min(system_rss),
                "max": max(system_rss),
                "avg": sum(system_rss) / len(system_rss),
            },
            "num_snapshots": len(self.history),
        }
    
    def save(self, path: str):
        """Save memory history to JSON."""
        with open(path, "w") as f:
            json.dump({
                "history": self.history,
                "summary": self.get_summary(),
            }, f, indent=2)
        print(f"Memory tracking saved to {path}")
    
    def print_current(self):
        """Print current memory usage."""
        gpu = self.get_gpu_memory()
        sys = self.get_system_memory()
        
        print("\n" + "="*60)
        print("MEMORY USAGE")
        print("="*60)
        if self.device == "cuda":
            print(f"GPU Memory:")
            print(f"  Allocated:     {gpu['allocated']:.1f} MB")
            print(f"  Reserved:      {gpu['reserved']:.1f} MB")
            print(f"  Peak:          {gpu['max_allocated']:.1f} MB")
        elif self.device == "mps":
            print(f"GPU Memory: MPS backend (detailed stats not available)")
        print(f"System Memory:")
        print(f"  RSS:           {sys['rss']:.1f} MB")
        print(f"  Process:       {sys['percent']:.1f}%")
        print("="*60 + "\n")


class MemoryTrackingCallback(TrainerCallback):
    """Callback to track memory during training."""
    
    def __init__(self, device: str = "cpu", log_every_n_steps: int = 10):
        self.tracker = MemoryTracker(device)
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize tracking."""
        self.start_time = time.time()
        print("\n🔍 Memory tracking enabled")
        self.tracker.print_current()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log memory after each step."""
        if state.global_step % self.log_every_n_steps == 0:
            loss = state.log_history[-1].get("loss") if state.log_history else None
            self.tracker.snapshot(step=state.global_step, loss=loss)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save final memory report."""
        elapsed = time.time() - self.start_time
        
        # Print summary
        print("\n" + "="*60)
        print("MEMORY TRACKING SUMMARY")
        print("="*60)
        summary = self.tracker.get_summary()
        
        if summary:
            print(f"Training time: {elapsed:.1f}s")
            print(f"\nGPU Memory (MB):")
            print(f"  Min:  {summary['gpu_memory_mb']['min']:.1f}")
            print(f"  Max:  {summary['gpu_memory_mb']['max']:.1f}")
            print(f"  Avg:  {summary['gpu_memory_mb']['avg']:.1f}")
            print(f"\nSystem Memory (MB):")
            print(f"  Min:  {summary['system_memory_mb']['min']:.1f}")
            print(f"  Max:  {summary['system_memory_mb']['max']:.1f}")
            print(f"  Avg:  {summary['system_memory_mb']['avg']:.1f}")
            print(f"\nSnapshots taken: {summary['num_snapshots']}")
        
        # Save to file
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "memory_tracking.json")
        self.tracker.save(save_path)
        
        print("="*60 + "\n")


def visualize_memory_tracking(json_path: str, output_path: Optional[str] = None):
    """
    Visualize memory tracking from saved JSON.
    
    Args:
        json_path: Path to memory_tracking.json
        output_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    history = data["history"]
    if not history:
        print("No memory history found")
        return
    
    steps = [h["step"] for h in history]
    gpu_mem = [h["gpu"]["allocated"] for h in history]
    sys_mem = [h["system"]["rss"] for h in history]
    losses = [h.get("loss") for h in history if "loss" in h]
    loss_steps = [h["step"] for h in history if "loss" in h]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Memory usage
    ax1.plot(steps, sys_mem, 'b-', label='System RAM', linewidth=2)
    if max(gpu_mem) > 0:
        ax1.plot(steps, gpu_mem, 'r-', label='GPU Memory', linewidth=2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Memory Usage During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    if losses:
        ax2.plot(loss_steps, losses, 'g-', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Memory plot saved to {output_path}")
    else:
        output_path = json_path.replace('.json', '_plot.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Memory plot saved to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example: visualize existing tracking data
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        visualize_memory_tracking(json_path)
    else:
        print("Usage: python memory_tracking.py <path_to_memory_tracking.json>")
