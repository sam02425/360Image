# Python Docstring Guide

Consistent docstrings make code easier to understand and maintain. We use Google-style docstrings across modules, functions, and classes.

## Module Docstrings
- Place a brief overview at the top of each Python file describing its purpose, primary entry points, and key dependencies.

Example:
```python
"""
Train YOLOv8 on the unified retail dataset and log metrics to MLflow.

Entry points:
- parse_args: CLI flags for dataset, epochs, device
- main: orchestrates training and logging
Dependencies: torch, ultralytics (lazy-loaded in main)
"""
```

## Function Docstrings (Google Style)
- Begin with a one-line summary, then an optional extended description.
- Document `Args`, `Returns`, and `Raises` as applicable.

Example:
```python
def parse_args():
    """Parse command-line arguments for YOLOv8 training.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    # implementation
```

Another example:
```python
def train(model, data, epochs):
    """Train a model on the provided dataset.

    Args:
        model: Initialized model instance.
        data (str | Path): Path to dataset YAML.
        epochs (int): Number of training epochs.

    Returns:
        dict: Training summary with key metrics.

    Raises:
        ValueError: If `epochs` is non-positive.
    """
    # implementation
```

## Class Docstrings
```python
class RetailDataset:
    """Loads and prepares retail images and labels for training.

    Attributes:
        root (Path): Dataset root directory.
        transforms (callable): Optional transforms applied per sample.
    """
    # implementation
```

## Style Notes
- Keep summaries to one sentence; elaborate only if helpful.
- Avoid repeating obvious details already evident from the code.
- Keep docstrings truthful and up-to-date with behavior.
- Prefer explicit types in `Args` and `Returns` where helpful.

## Checklist
- Module-level docstring present and accurate
- Public functions/classes have docstrings
- Docstrings reflect current functionality
- Cross-references to related modules/scripts included where useful