# tests/test_core.py

import pytest
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Import the public API 'watch'
from gradlens import watch 
from gradlens.core import Monitor
from gradlens.state import State
from gradlens.hooks import HookManager

@pytest.fixture
def simple_model():
    """Provides a basic nn.Module for testing."""
    return nn.Linear(10, 5)

def test_watch_initialization(simple_model):
    """
    Tests if `gl.watch()` correctly creates a Monitor
    and that the Monitor initializes its components.
    """
    # We patch `HookManager.attach_hooks` to verify it gets called
    # by the Monitor's __init__.
    with patch.object(HookManager, 'attach_hooks') as mock_attach:
        
        monitor = watch(simple_model)

        assert isinstance(monitor, Monitor)
        assert isinstance(monitor.state, State)
        assert isinstance(monitor.hooks, HookManager)
        
        # Check that __init__ correctly called attach_hooks
        mock_attach.assert_called_once()
        assert monitor.is_active is True

def test_monitor_log_calls_state_methods(simple_model):
    """
    Tests that `monitor.log()` correctly orchestrates the state.
    
    This is a critical unit test. We mock the state object to
    ensure the Monitor *only* calls the correct methods
    without worrying about what those methods *do*.
    """
    monitor = watch(simple_model)
    
    # Replace the real state with a MagicMock
    monitor.state = MagicMock(spec=State)
    
    # --- Act ---
    test_loss = 0.123
    monitor.log(loss=test_loss)

    # --- Assert ---
    # 1. Did we log the manual loss?
    monitor.state.log_loss.assert_called_once_with(test_loss)
    
    # 2. Did we trigger the data processing/commit?
    monitor.state.process_hook_data.assert_called_once()

def test_get_stats_and_history_calls_state(simple_model):
    """
    Tests that the data retrieval methods (`get_stats`, `get_history`)
    correctly fetch data from the state.
    """
    monitor = watch(simple_model)
    
    # Replace state with a mock
    monitor.state = MagicMock(spec=State)

    # --- Test get_stats ---
    mock_stats = {"loss": 0.1}
    monitor.state.get_current_stats.return_value = mock_stats
    
    stats = monitor.get_stats()
    monitor.state.get_current_stats.assert_called_once()
    assert stats == mock_stats
    
    # --- Test get_history ---
    mock_history = {"loss": [0.5, 0.1]}
    monitor.state.get_full_history.return_value = mock_history
    
    history = monitor.get_history()
    monitor.state.get_full_history.assert_called_once()
    assert history == mock_history

def test_monitor_cleanup_with_context_manager(simple_model):
    """
    Tests that the monitor automatically cleans up hooks
    when used as a context manager (i.e., `with` statement).
    """
    # Patch the *cleanup* method to see if it gets called
    with patch.object(Monitor, '_cleanup') as mock_cleanup:
        
        with watch(simple_model) as monitor:
            assert isinstance(monitor, Monitor)
            # Cleanup should not be called *inside* the block
            mock_cleanup.assert_not_called()
            
        # Cleanup *must* be called after exiting the block
        mock_cleanup.assert_called_once()

def test_monitor_close_method(simple_model):
    """
    Tests that the manual `.close()` method also triggers cleanup.
    """
    with patch.object(Monitor, '_cleanup') as mock_cleanup:
        
        monitor = watch(simple_model)
        mock_cleanup.assert_not_called() # Not called on init
        
        monitor.close()
        mock_cleanup.assert_called_once()

def test_monitor_log_with_custom_metrics(simple_model):
    """
    Tests that monitor.log() can accept and record
    additional, user-defined metrics like 'accuracy'.
    """
    monitor = watch(simple_model)
    
    # --- Act ---
    test_loss = 0.5
    custom_metrics = {"accuracy": 0.95, "f1_score": 0.8}
    
    monitor.log(loss=test_loss, metrics=custom_metrics)
    
    # --- Assert ---
    stats = monitor.get_stats()
    history = monitor.get_history()

    # Check that metrics were recorded correctly in the last step
    assert stats["loss"] == test_loss
    assert stats["custom_metrics"] == custom_metrics
    
    # Check that history was appended
    assert len(history["custom_metrics"]) == 1
    assert history["custom_metrics"][0] == custom_metrics