# src/gradlens/hooks.py

import torch
import torch.nn as nn
from typing import List, Any, Callable, Optional, Set

# We import our new, simple State class
from .state import State 

class HookManager:
    """
    Manages the attachment and removal of PyTorch hooks.

    This class is responsible for the "magic" of data collection.
    It attaches lightweight hooks to the model's parameters and
    modules to automatically collect data during the backward pass
    and writes that data directly into the provided State object.
    """

    def __init__(self, model: nn.Module, state: State):
        """
        Initializes the HookManager.

        Args:
            model: The PyTorch model (nn.Module) to attach hooks to.
            state: The central State object where data will be recorded.
        """
        self.model = model
        self.state = state
        
        # This list will store all active hook handles
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        
        # Keep track of attached modules to avoid duplicates
        self._attached_modules: Set[nn.Module] = set()

    def attach_hooks(self) -> None:
        """
        Attaches all necessary hooks to the model for v0.1 metrics.
        
        - Attaches a backward hook to all `requires_grad` parameters for Grad Norms.
        - Attaches a forward hook to all `nn.ReLU` modules for Dead Neuron detection.
        """
        if self.handles:
            # Prevent double-attaching
            print("Warning: Hooks are already attached.")
            return

        # 1. Attach Gradient Hooks (Backward Hook)
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(
                    self._create_backward_hook(param_name)
                )
                self.handles.append(handle)

        # 2. Attach Dead Neuron Hooks (Forward Hook)
        for layer_name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                # Only attach to unique ReLU modules
                if module not in self._attached_modules:
                    handle = module.register_forward_hook(
                        self._create_forward_hook(layer_name)
                    )
                    self.handles.append(handle)
                    self._attached_modules.add(module)

    def _create_backward_hook(self, param_name: str) -> Callable:
        """
        Factory function to create the backward hook for a specific parameter.
        This closure captures the `param_name`.
        
        This hook fires *after* gradients have been computed and accumulated.
        """
        def hook(param: torch.Tensor) -> None:
            # param.grad is the accumulated gradient
            if param.grad is None:
                return

            grad = param.grad.detach()

            # --- Health Check ---
            if not torch.isfinite(grad).all():
                # A NaN/Inf gradient is a catastrophic event.
                # We record it in the state and stop calculation.
                self.state.record_nan_alert(f"Gradient@{param_name}")
                return

            # --- v0.1 Metric: Gradient Norm ---
            try:
                # Calculate L2 norm
                norm = torch.norm(grad).item()
                self.state.record_grad_norm(param_name, norm)
            except Exception as e:
                # Fail gracefully if norm calculation fails
                self.state.record_grad_norm(param_name, float('nan'))

        return hook

    def _create_forward_hook(self, layer_name: str) -> Callable:
        """
        Factory function to create the forward hook for a nn.ReLU module.
        This closure captures the `layer_name`.
        
        This hook fires *after* the module's forward pass.
        """
        def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
            # We only expect a single tensor output from a ReLU
            if not isinstance(outputs, torch.Tensor):
                return
                
            output = outputs.detach()

            # --- v0.1 Metric: Dead Neuron Percentage ---
            try:
                # A "dead neuron" in a ReLU is one that outputs 0.
                dead_pct = (output == 0).float().mean().item()
                self.state.record_dead_neurons(layer_name, dead_pct)
            except Exception as e:
                # Fail gracefully
                self.state.record_dead_neurons(layer_name, float('nan'))

        return hook

    def remove_hooks(self) -> None:
        """
        Removes all registered hooks from the model.
        """
        for handle in self.handles:
            handle.remove()
        
        # Clear the lists to prevent memory leaks and re-runs
        self.handles.clear()
        self._attached_modules.clear()

    # It should call `remove_hooks` directly.
    def close(self) -> None:
        """
        Manually removes hooks and stops monitoring.
        Alias for remove_hooks() for a cleaner public API.
        """
        self.remove_hooks()

    def __del__(self) -> None:
        """
        Destructor to ensure hooks are removed when the object is
        garbage collected, preventing memory leaks.
        """
        if self.handles:
            # We print a warning because explicit .close() or 'with' block
            # is the preferred way to clean up.
            print("Warning: GradLens HookManager was not explicitly closed. "
                  "Cleaning up hooks to prevent memory leaks.")
            self.remove_hooks()