import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ==============================
# 1️⃣ ĐỊNH NGHĨA MÔ HÌNH SIMPLECNN ĐÚNG VỚI TRỌNG SỐ ĐÃ LƯU
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):  # nếu bạn train 6 lớp, giữ nguyên
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 150x150 → 75x75 → 37x37 → 18x18
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 18 * 18, 256),   # đúng với 41472 input features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ==============================
# 2️⃣ LOAD MÔ HÌNH
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "simplecnn_model.pth"

model = SimpleCNN(num_classes=6).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Mô hình SimpleCNN đã được load thành công!")

# ==============================
# 3️⃣ TIỀN XỬ LÝ ẢNH
# ==============================
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # đúng như lúc train
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dir = r"D:\STUDY\HK1-25-26\thi-giac-may-tinh\cnn\pred"

# ==============================
# 4️⃣ DỰ ĐOÁN ẢNH
# ==============================
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # theo dataset Intel Image Classification

for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            _, pred = torch.max(outputs, 1)
            pred_class = classes[pred.item()]

        print(f"🖼️ {filename} → {pred_class}")
