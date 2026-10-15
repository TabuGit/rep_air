import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

img = Image.open("image.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)

probs = torch.nn.functional.softmax(out[0], dim=0)
top5_prob, top5_idx = torch.topk(probs, 5)
categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

for i in range(5):
    print(f"{i+1}. {categories[top5_idx[i]]}: {top5_prob[i].item()*100:.2f}%")