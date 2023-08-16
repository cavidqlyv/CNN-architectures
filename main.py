#%%
import numpy as np
import torch
import torch.nn as nn
from VGG16 import VGG16
from utils import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, valid_loader = data_loader('./data', 64)
test_loader = data_loader("data",64, test=True)

# Hyper Parameters
num_classes = 100
num_epochs = 20
batch_size = 16
learning_rate = 0.001
model = VGG16(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
total_step

print(model.parameters)
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        output = model(images)
        loss = criterion(output, labels)

        # Backward Porop and optimizer 
        optimizer.zero_grad()  # Wt1 = Wt0 - LR * Gradient
        loss.backward() # backward
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss {loss.item():.4f}")

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _ , predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            del images, labels, outputs
            
    print(f"Accuracy of the model on {total} validation images: {(correct/total)*100}")
