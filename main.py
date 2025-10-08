# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("puneet6060/intel-image-classification")

# print("Path to dataset files:", path)



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time


data_dir = r"D:\STUDY\HK1-25-26\thi-giac-may-tinh\cnn\intel-image-classification" 

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(root=f"{data_dir}/seg_train", transform=transform)
test_data = datasets.ImageFolder(root=f"{data_dir}/seg_test", transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        # ---- Feature extraction ----
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # Pool1
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # Pool2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# Conv3
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # Pool3
        )
        
        # ---- Classification ----
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 18 * 18, 256),  # fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)      # output layer (Softmax implicit)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x

#khởi tạo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=6).to(device)
print(model)  # In kiến trúc mô hình

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# HUẤN LUYỆN VÀ ĐÁNH GIÁ

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    # Evaluate
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    # Lưu lại lịch sử
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
