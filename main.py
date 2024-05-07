import torch
from torchvision import datasets, transforms

# Transformation definieren, um Bilder entsprechend anzupassen
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Größe der Bilder ändern
    transforms.ToTensor()            # Konvertierung in Tensor
])

# Dataset erstellen
dataset = datasets.ImageFolder(root='archive/train', transform=transform)

# DataLoader erstellen
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

from torchvision import models
import torch.nn as nn

# Modell laden und anpassen
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 53)  # 53 Klassen

# Überprüfen, ob GPU verfügbar ist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trainingsloop
for epoch in range(10):  # Anzahl der Epochen
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'karten_erkenner_model.pth')

