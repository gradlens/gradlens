# SPDX-FileCopyrightText: 2025-present im-mahdi-74 <mahdi.mosavi.nsa@gmail.com>
#
# SPDX-License-Identifier: MIT

# src/gradlens/__init__.py

"""
GradLens: An enterprise-grade PyTorch monitoring and analytics toolkit.
"""

import torch.nn as nn
from .core import Monitor  # Import the main Monitor class
from .__about__ import __version__ # Get version from hatch's file

# --- Public API ---

def watch(model: nn.Module) -> Monitor:
    """
    Initializes the GradLens monitor for a given PyTorch model.

    This is the main entry point for the library. It attaches
    the necessary hooks and returns a Monitor instance.

    Args:
        model: The PyTorch model (nn.Module) to monitor.

    Returns:
        A Monitor instance ready for logging.

    Example:
        >>> import gradlens as gl
        >>> model = YourModel()
        >>> with gl.watch(model) as monitor:
        >>>     # ... training loop ...
        >>>     monitor.log(loss)
    """
    # All complexity is hidden within the Monitor class
    return Monitor(model)

# --- Expose main components ---
__all__ = [
    "watch",
    "Monitor",
    "__version__"
]