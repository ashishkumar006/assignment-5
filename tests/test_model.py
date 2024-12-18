import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import CompactMNIST
from train import train_one_epoch, count_parameters

def test_parameter_count():
    model = CompactMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_training_accuracy():
    accuracy = train_one_epoch()
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be greater than 95%" 