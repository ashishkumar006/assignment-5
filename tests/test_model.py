import torch
import pytest
from model import CompactMNIST
from train import train_one_epoch, count_parameters

def test_parameter_count():
    model = CompactMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25,000"

def test_model_accuracy():
    accuracy = train_one_epoch()
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below the required 95%"