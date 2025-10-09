import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =========================
# 1️⃣ Chuẩn bị dữ liệu
# =========================
train_dir = r"D:\STUDY\HK1-25-26\thi-giac-may-tinh\cnn\intel-image-classification\seg_train"
test_dir  = r"D:\STUDY\HK1-25-26\thi-giac-may-tinh\cnn\intel-image-classification\seg_test"

transform = {
    'train': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ]),
    'test': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
}

train_data = datasets.ImageFolder(train_dir, transform=transform['train'])
test_data  = datasets.ImageFolder(test_dir,  transform=transform['test'])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"🧾 Số lượng ảnh train: {len(train_data)} | test: {len(test_data)}")

# =========================
# 2️⃣ Định nghĩa AlexNet
# =========================
class AlexNet(nn.Module):
    def __init__(self, num_classes=6):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =========================
# 3️⃣ Huấn luyện mô hình
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=6).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []

for epoch in range(num_epochs):
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # Đánh giá trên tập test
    model.eval()
    test_loss, correct, total = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    test_acc = 100 * correct / total
    test_loss /= len(test_loader)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

    print(f"📅 Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# =========================
# 💾  Lưu mô hình sau khi huấn luyện
# =========================
torch.save(model.state_dict(), "alexnet_model.pth")
print("✅ Mô hình đã được lưu thành công vào file alexnet_model.pth")

# =========================
# 4️⃣ Vẽ biểu đồ Loss & Accuracy
# =========================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Độ chính xác (Accuracy) qua các Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Hàm mất mát (Loss) qua các Epoch')
plt.legend()
plt.show()

# =========================
# 5️⃣ Chỉ số đánh giá chi tiết
# =========================
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

# Ma trận nhầm lẫn
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🧩 Ma trận nhầm lẫn (Confusion Matrix)")
plt.show()

# =========================
# 6️⃣ Bảng kiến trúc AlexNet
# =========================
layers = [
    ["In", "Input", "3 (RGB)", "227×227", "-", "-", "-", "-"],
    ["C1", "Convolution", "96", "55×55", "11×11", "4", "valid", "ReLU"],
    ["S2", "Max Pooling", "96", "27×27", "3×3", "2", "valid", "-"],
    ["C3", "Convolution", "256", "27×27", "5×5", "1", "same", "ReLU"],
    ["S4", "Max Pooling", "256", "13×13", "3×3", "2", "valid", "-"],
    ["C5", "Convolution", "384", "13×13", "3×3", "1", "same", "ReLU"],
    ["C6", "Convolution", "384", "13×13", "3×3", "1", "same", "ReLU"],
    ["C7", "Convolution", "256", "13×13", "3×3", "1", "same", "ReLU"],
    ["F8", "Fully connected", "-", "-", "-", "-", "-", "ReLU"],
    ["F9", "Fully connected", "-", "-", "-", "-", "-", "ReLU"],
    ["Out", "Fully connected", "-", "-", "-", "-", "-", "Softmax"]
]

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=layers,
                 colLabels=["Layer", "Type", "Maps", "Size", "Kernel size", "Stride", "Padding", "Activation"],
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("📘 Cấu trúc mô hình AlexNet", fontsize=14)
plt.show()
