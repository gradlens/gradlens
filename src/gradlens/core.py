# src/gradlens/core.py

import torch.nn as nn
from typing import Dict, Any

from .state import State
from .hooks import HookManager

from typing import Dict, Any, Optional


class Monitor:
    """
    The main monitoring class that orchestrates state and hooks.

    This class is instantiated via `gradlens.watch()`. It manages
    the lifecycle of hooks and provides the user-facing API
    (e.g., .log(), .get_stats()).
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the Monitor.

        Args:
            model: The PyTorch model to be monitored.
        """
        self.model = model
        self.state = State()  # The central data store
        self.hooks = HookManager(self.model, self.state)  # The hook manager

        # Automatically attach hooks on initialization
        try:
            self.hooks.attach_hooks()
            self.is_active = True
        except Exception as e:
            # Handle potential hook attachment failures
            self.is_active = False
            print(f"Error: Failed to initialize GradLens Monitor. {e}")
            # We might want a more robust logging/warning here
            
    def log(self, loss: float, metrics: Optional[Dict[str, float]] = None) -> None:        
        """
        Logs a single training step's loss.

        This method should be called once per training step.
        It records the provided loss and triggers the processing
        of all data collected by the hooks for this step.

        Args:
            loss: The scalar loss value for the current step.
        """
        if not self.is_active:
            return  # Do nothing if initialization failed

        # 1. Log the manually provided loss
        self.state.log_loss(loss)

        if metrics:
            self.state.log_custom_metrics(metrics)

        # 2. Process all data collected by hooks in this step
        # (e.g., aggregate gradient norms, clear buffers)
        self.state.process_hook_data()

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves the statistics for the *last logged step*.

        Returns:
            A dictionary containing the latest computed stats, e.g.:
            {
                'loss': 0.123,
                'grad_norm': {'layer1.weight': 0.5, ...},
                'dead_neuron_pct': {'relu1': 0.2, ...}
            }
        """
        if not self.is_active:
            return {}
        return self.state.get_current_stats()

    def get_history(self) -> Dict[str, Any]:
        """
        Retrieves the complete history of all stats since monitoring began.

        Returns:
            A dictionary containing lists of all historical stats.
        """
        if not self.is_active:
            return {}
        return self.state.get_full_history()

    def _cleanup(self) -> None:
        """Safely removes all attached hooks."""
        if self.is_active:
            self.hooks.remove_hooks()
            self.is_active = False

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        self._cleanup()

# --- Context Manager Interface (`with gl.watch(...) as monitor:`) ---

    def __enter__(self) -> "Monitor":
        """Allows the Monitor to be used as a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Cleans up hooks upon exiting the 'with' block.
        """
        self._cleanup() 

    def close(self) -> None:
        """Manually remove hooks and stop monitoring."""
        self._cleanup()