import torch
from model import Model
from torchvision import transforms
from PIL import Image

model = Model()
model.load_state_dict(torch.load("levit_model.pth"))
model.eval()

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return output.item()
