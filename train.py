import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CompactMNIST
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch():
    # Enhanced transforms for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,  # Increased rotation
                translate=(0.1, 0.1),  # Increased translation
                scale=(0.9, 1.1),  # Increased scale variation
                fill=0
            )
        ], p=0.7)  # Increased probability
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactMNIST().to(device)
    
    param_count = count_parameters(model)
    print(f"Model has {param_count} parameters")
    if param_count >= 25000:
        raise ValueError("Model has too many parameters!")

    criterion = nn.NLLLoss()  # Changed to NLLLoss since we're using log_softmax
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # Reduced initial learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=1e-6
    )

    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        running_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}] | '
                  f'Loss: {running_loss / (batch_idx + 1):.3f} | '
                  f'Acc: {100.*correct/total:.2f}%')

    accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {accuracy:.2f}%')
    
    if accuracy < 95:
        raise ValueError("Model accuracy is below 95%!")
    
    return accuracy

if __name__ == "__main__":
    train_one_epoch() 