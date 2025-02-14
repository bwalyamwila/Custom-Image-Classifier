import torch
import torchvision
from PIL import Image
import numpy as np
from collections import OrderedDict
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os, random
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import device
from tempfile import TemporaryDirectory
import time

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

mean = np.array([0.485, 0.456, 0.406])
stdv = np.array([0.229, 0.224, 0.225])

data_transforms = data_transforms = {   
  'train' : transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,stdv)
                                      ]),
  
  'valid' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,stdv)
                                      ]),

  'test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, stdv)
                                      ]),
    
}

image_datasets = image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle=True, num_workers=4)
                  for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load pre-trained VGG19 model
model_ft = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

classifier = nn.Sequential(
    nn.Linear(model_ft.classifier[0].in_features, 512),  
    nn.ReLU(),
    nn.Dropout(p=0.5),  
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 102)  
)

model_ft.classifier = classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
epochs = 8

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs = epochs):
    since = time.time()
    best_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    
    model.load_state_dict(best_model_wts)

    return model


model_ft.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)


accuracy = 0
pass_count = 0

with torch.no_grad():
    for images, labels in dataloaders['test']:  
        pass_count += 1

        images, labels = images.to(device), labels.to(device)

        outputs = model_ft(images)
        ps = torch.exp(outputs)  

        equality = (labels == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()


print(f"Testing Accuracy: {accuracy / pass_count:.4f}")

model_ft.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg19',
              'learning_rate': 0.001,
              'batch_size': 32,
              'classifier' : classifier,
              'epochs': epochs,
              'optimizer': optimizer_ft.state_dict(),
              'state_dict': model_ft.state_dict(),
              'class_to_idx': model_ft.class_to_idx}

torch.save(checkpoint, 'final_model.pth')

def load_checkpoint(filename):
    checkpoint = torch.load(filename)

    model_ft = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    for param in model_ft.parameters():
        param.requires_grad = False
        
    model_ft.classifier = checkpoint['classifier']
    model_ft.load_state_dict(checkpoint['state_dict'])
    model_ft.class_to_idx = checkpoint['class_to_idx']
    optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=checkpoint['learning_rate'])
    
    return model_ft, optimizer_ft

nn_filename = 'final_model.pth'

model, optimizer = load_checkpoint(nn_filename)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    image = Image.open(image_path)
    
    # Resize and crop the image
    image = image.resize((256, 256))
    value = 0.5 * (256 - 224)
    image = image.crop((value, value, 256 - value, 256 - value))
    
    # Convert the image to a NumPy array and normalize
    image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()  
    
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
    else:
        model.cpu()
        print("We go for CPU")
    
    model.eval()

    image = process_image(image_path)
    
    image = torch.from_numpy(np.array([image])).float()
    
    image = Variable(image)
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0] 
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label