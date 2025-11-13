# tests/test_hooks.py

import torch
import torch.nn as nn
import pytest

# Import our components from the 'src' layout
from gradlens.state import State
from gradlens.hooks import HookManager

# --- Test Fixture ---

@pytest.fixture
def simple_model_and_state():
    """
    Provides a simple model, a state object, and dummy data
    for each test.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()  # We will hook this
            self.layer2 = nn.Linear(20, 1)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    model = SimpleModel()
    state = State()
    # Dummy input data
    x = torch.randn(4, 10)
    return model, state, x

# --- Test Cases ---

def test_grad_norm_hook(simple_model_and_state):
    """
    Tests if the backward hook correctly records gradient norms.
    """
    # 1. Setup
    model, state, x = simple_model_and_state
    hook_manager = HookManager(model, state)
    hook_manager.attach_hooks()

    # 2. Execute
    loss = model(x).sum()
    loss.backward()  # This triggers the backward hooks

    # Manually process the data (Monitor.log() would do this)
    state.process_hook_data()
    history = state.get_full_history()

    # 3. Assert
    assert len(history['grad_norm']) == 1  # One step was processed
    
    grad_norms = history['grad_norm'][0]
    assert 'layer1.weight' in grad_norms
    assert 'layer1.bias' in grad_norms
    assert 'layer2.weight' in grad_norms
    assert 'layer2.bias' in grad_norms
    
    # Check that norms are positive floats
    assert grad_norms['layer1.weight'] > 0
    assert isinstance(grad_norms['layer1.weight'], float)
    
    # Check that hooks were removed (optional, good practice)
    hook_manager.remove_hooks()
    assert len(hook_manager.handles) == 0

def test_dead_neuron_hook(simple_model_and_state):
    """
    Tests if the forward hook correctly identifies dead neurons.
    """
    # 1. Setup
    model, state, _ = simple_model_and_state
    
    # Force the bias of the first linear layer to be very negative
    # This guarantees that the input to ReLU will be negative
    with torch.no_grad():
        model.layer1.bias.fill_(-10.0)

    hook_manager = HookManager(model, state)
    hook_manager.attach_hooks()

    # 2. Execute
    # Create an input that won't overcome the negative bias
    x_zeros = torch.zeros(4, 10)
    
    # Forward pass triggers the forward hook on nn.ReLU
    model(x_zeros)

    # Manually process
    state.process_hook_data()
    history = state.get_full_history()

    # 3. Assert
    assert len(history['dead_neuron_pct']) == 1
    
    dead_neurons = history['dead_neuron_pct'][0]
    # 'relu' is the name of our module in SimpleModel
    assert 'relu' in dead_neurons
    
    # Since bias was -10 and input was 0,
    # 100% of neurons in the ReLU layer should be "dead" (output 0)
    assert dead_neurons['relu'] == 1.0

def test_nan_gradient_handling(simple_model_and_state):
    """
    Tests if the backward hook correctly detects a NaN gradient
    and logs an alert in the State.
    """
    # 1. Setup
    model, state, x = simple_model_and_state
    hook_manager = HookManager(model, state)
    hook_manager.attach_hooks()

    # 2. Execute
    # Create a "bad" loss (NaN)
    output = model(x)
    loss = output.sum() * torch.tensor(float('nan'))

    # We expect PyTorch to warn about non-finite grads,
    # but our hook should catch it.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss.backward()

    # Manually process
    state.process_hook_data()
    history = state.get_full_history()

    # 3. Assert
    assert len(history['nan_alerts']) == 1
    
    # Check that an alert was logged
    alert = history['nan_alerts'][0]
    assert alert is not None
    assert 'Gradient@' in alert # e.g., 'Gradient@layer2.weight'
    
    # Check that grad norm calculation was skipped
    grad_norms = history['grad_norm'][0]
    assert len(grad_norms) == 0 # No norms were recorded