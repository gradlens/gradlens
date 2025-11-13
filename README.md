# 🔥 GradLens

[![CI](https://github.com/gradlens/gradlens/actions/workflows/ci.yml/badge.svg)](https://github.com/gradlens/gradlens/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/gradlens.svg)](https://badge.fury.io/py/gradlens)


An enterprise-grade **PyTorch monitoring and analytics toolkit.**

---

## 🚀 Quick Start (v0.1)

`gradlens` provides a lightweight, zero-overhead monitoring engine for your PyTorch training loops.

---

### Installation

```bash
pip install gradlens
```

---

### Usage

Use the `gl.watch()` context manager to automatically attach hooks to your model. The `monitor.log()` method captures statistics with near-zero performance impact.

```python
import torch
import torch.nn as nn
import gradlens as gl  # Import the library

# 1. Define your model and data (example)
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
data = torch.randn(16, 10)
target = torch.randn(16, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 2. Wrap your training loop with gl.watch()
try:
    with gl.watch(model) as monitor:
        for _ in range(10):  # Example training loop

            # --- Training Step ---
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # Hooks capture data here
            optimizer.step()

            # 3. Log the loss
            monitor.log(loss=loss.item())

except Exception as e:
    print(f"Training failed: {e}")

finally:
    # 4. Get the results
    history = monitor.get_history()

    print("\n--- Training History ---")
    print(f"Total steps: {len(history['loss'])}")
    print(f"Final loss: {history['loss'][-1]}")

    # Show stats for the last step
    last_stats = monitor.get_stats()
    print("\n--- Last Step Stats ---")
    print(f"Gradient Norms: {last_stats['grad_norm']}")
    print(f"Dead Neuron %: {last_stats['dead_neuron_pct']}")
```

---

This is the **v0.1 engine**. Visualization and advanced analytics will be added in future versions.

---

## 🧑‍💻 Development & Packaging

1. **Set up a virtual environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip hatch
   ```
2. **Run the automated test suite**
   ```bash
   hatch run test
   ```
3. **Build distributable artifacts (wheel + sdist)**
   ```bash
   HATCH_HOME=.hatch hatch build
   ```
   > Setting `HATCH_HOME` keeps Hatch’s build env inside the repo, which works well on locked-down machines/CI runners.
4. **Publish to PyPI (manual invocation)**
   ```bash
   HATCH_INDEX_USER="__token__" \
   HATCH_INDEX_AUTH="$PYPI_API_TOKEN" \
   hatch publish pypi
   ```

Tagging releases as `v*` lets GitHub Actions (when re-enabled) run the same build/publish pipeline automatically.
