# Compact MNIST Classifier

This repository contains a PyTorch implementation of a compact MNIST classifier that achieves >95% accuracy in a single epoch while using less than 25,000 parameters.

## Model Architecture

The model uses an efficient CNN architecture with:
- Initial convolution (12 filters)
- Residual block with projection shortcut (16 filters)
- Final convolution (24 filters)
- Batch normalization throughout
- Single fully connected layer
- Approximately 24,000 parameters

Key features:
- Residual connections for better gradient flow
- Batch normalization for training stability
- Efficient parameter usage
- Dropout for regularization

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest (for testing)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train.py
```

## Testing

The repository includes GitHub Actions that automatically test:
1. The model parameter count (must be <25,000)
2. Training accuracy (must be >95% in one epoch)

To run tests locally:

```bash
pytest tests/
```

## Model Design Choices

- Uses residual connections for better gradient flow
- Employs batch normalization for faster convergence
- Carefully balanced channel progression (1→12→16→24)
- Efficient parameter usage through modern architectural choices
- Dropout and data augmentation for regularization

## License

MIT