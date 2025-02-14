import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import time
import os
import model
from model import (
    data_transforms, image_datasets, dataloaders, dataset_sizes, class_names,
    train_model, device, model_ft, optimizer_ft, criterion, exp_lr_scheduler
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
    parser.add_argument('data_directory', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19', help='Model architecture (e.g., vgg19)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data_dir = args.data_directory
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms and datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes

    # Build model
    print("Building model...")
    if args.arch == 'vgg19':
        model_ft = models.vgg19(pretrained=True)
        input_size = 25088
    elif args.arch == 'vgg16':
        model_ft = models.vgg16(pretrained=True) 
        input_size = 25088
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024 
    else:
        raise ValueError("Unsupported architecture. Use 'vgg19'.")

    for param in model_ft.parameters():
        param.requires_grad = False

    # Define classifier
    classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model_ft.classifier = classifier

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

    model_ft = model_ft.to(device)

    # Train model
    print("Training model...")
    model_ft = train_model(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=args.epochs
    )

    # Save checkpoint
    print("Saving checkpoint...")
    checkpoint = {
        'input_size': 25088,
        'output_size': 102,
        'arch': args.arch,
        'learning_rate': args.learning_rate,
        'batch_size': 32,
        'classifier': classifier,
        'epochs': args.epochs,
        'optimizer': optimizer_ft.state_dict(),
        'state_dict': model_ft.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx
    }

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    main()