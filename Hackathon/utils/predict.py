import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# load model
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

# load image
img = Image.open("test.png").convert("L")
arr = np.array(img)

# binarize
arr = (arr > 128).astype(np.uint8) * 255

# crop
coords = np.column_stack(np.where(arr > 0))
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0)
arr = arr[y0:y1, x0:x1]

# resize to 20x20
img = Image.fromarray(arr)
img = img.resize((20, 20))

# center in 28x28
new_img = Image.new("L", (28, 28))
new_img.paste(img, (4, 4))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

x = transform(new_img).unsqueeze(0)

# predict
with torch.no_grad():
    output = model(x)
    pred = torch.argmax(output, dim=1)

if pred.item() == 7:
    # check if it's a thin vertical line → probably 1
    col_sum = new_img.getbbox()  # just to force using image

    arr_check = np.array(new_img)
    vertical_pixels = np.sum(arr_check > 200, axis=0)
    
    if max(vertical_pixels) > 20:  # strong vertical stroke
        pred = torch.tensor([1])




def predict_image(model, image_path, classes, device):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]        