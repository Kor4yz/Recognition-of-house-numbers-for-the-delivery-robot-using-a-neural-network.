import argparse, torch
from PIL import Image
import torchvision.transforms as T
from pytorch_lightning import LightningModule
from src.models.resnet import SVHNResNet

class LitWrap(LightningModule):
    def __init__(self): super().__init__(); self.net = SVHNResNet()
    def forward(self, x): return self.net(x)

def load_image(path: str):
    img = Image.open(path).convert("RGB").resize((32,32))
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4377,0.4438,0.4728], std=[0.1980,0.2010,0.1970])
    ])
    return tfm(img).unsqueeze(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--image_path", required=True)
    args = ap.parse_args()

    model = LitWrap().load_from_checkpoint(args.checkpoint_path, map_location="cpu")
    model.eval()
    x = load_image(args.image_path)
    with torch.inference_mode():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    print(f"Predicted digit: {pred}")
