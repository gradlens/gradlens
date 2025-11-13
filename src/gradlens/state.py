# src/gradlens/state.py

import torch
from typing import Dict, List, Any, Optional

class State:
    """
    A decoupled state container for managing training statistics.
    (Updated to handle NaN/Inf alerts)
    """

    def __init__(self):
        # --- 1. Step Buffers (Staging Area) ---
        self.current_loss: float = 0.0
        self.current_grad_norms: Dict[str, float] = {}
        self.current_dead_neurons: Dict[str, float] = {}
        # Buffer for alerts *within* the current step
        self.current_nan_alert: Optional[str] = None 
        self.current_custom_metrics: Dict[str, float] = {}
        # --- 2. Full History (Long-Term Storage) ---
        self.loss_history: List[float] = []
        self.grad_norm_history: List[Dict[str, float]] = []
        self.dead_neuron_history: List[Dict[str, float]] = []
        # History of alerts from *all* past steps
        self.nan_alert_history: List[Optional[str]] = []
        self.metrics_history: List[Dict[str, float]] = []

    # --- Methods for HookManager (Writers) ---

    def record_grad_norm(self, layer_name: str, norm: float) -> None:
        self.current_grad_norms[layer_name] = norm

    def record_dead_neurons(self, layer_name: str, pct: float) -> None:
        self.current_dead_neurons[layer_name] = pct
        
    def record_nan_alert(self, source: str) -> None:
        """
        Called by hooks if a NaN or Inf is detected.
        We only record the first alert per step to avoid spam.
        """
        if self.current_nan_alert is None:
            self.current_nan_alert = source

    # --- Methods for Monitor (Orchestrator & Readers) ---

    def log_loss(self, loss: float) -> None:
        self.current_loss = loss
        if not torch.isfinite(torch.tensor(loss)):
             self.record_nan_alert(f"Loss")

    def log_custom_metrics(self, metrics: Dict[str, float]) -> None:
            """Called by Monitor.log() to record custom metrics."""
            self.current_custom_metrics = metrics.copy()

    def process_hook_data(self) -> None:
        """
        "Commits" all data from the temporary step buffers
        into the permanent history lists and clears the buffers.
        """
        # 1. Commit current data to history
        self.loss_history.append(self.current_loss)
        self.grad_norm_history.append(self.current_grad_norms.copy())
        self.dead_neuron_history.append(self.current_dead_neurons.copy())
        self.nan_alert_history.append(self.current_nan_alert)
        self.metrics_history.append(self.current_custom_metrics.copy())

        # 2. Clear step buffers for the next iteration
        self.current_loss = 0.0
        self.current_grad_norms.clear()
        self.current_dead_neurons.clear()
        self.current_nan_alert = None # Reset alert
        self.current_custom_metrics.clear()

    def get_current_stats(self) -> Dict[str, Any]:
        return {
            "loss": self.loss_history[-1] if self.loss_history else None,
            "grad_norm": self.grad_norm_history[-1] if self.grad_norm_history else {},
            "dead_neuron_pct": self.dead_neuron_history[-1] if self.dead_neuron_history else {},
            "nan_alert": self.nan_alert_history[-1] if self.nan_alert_history else None,
            "custom_metrics": self.metrics_history[-1] if self.metrics_history else {},
        }

    def get_full_history(self) -> Dict[str, Any]:
        return {
            "loss": self.loss_history,
            "grad_norm": self.grad_norm_history,
            "dead_neuron_pct": self.dead_neuron_history,
            "nan_alerts": self.nan_alert_history,
            "custom_metrics": self.metrics_history,
        }