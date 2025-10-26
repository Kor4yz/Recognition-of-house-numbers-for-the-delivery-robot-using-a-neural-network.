import argparse, io, streamlit as st, torch
from PIL import Image
import torchvision.transforms as T
from pytorch_lightning import LightningModule
from src.models.resnet import SVHNResNet

class LitWrap(LightningModule):
    def __init__(self): super().__init__(); self.net = SVHNResNet()
    def forward(self, x): return self.net(x)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((32,32))
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4377,0.4438,0.4728], std=[0.1980,0.2010,0.1970])
    ])
    return tfm(img).unsqueeze(0)

def main(checkpoint_path: str):
    st.title("SVHN House Number Recognition")
    st.caption("Delivery robot helper • PyTorch Lightning")
    file = st.file_uploader("Upload an image with a digit", type=["png","jpg","jpeg"])
    if "model" not in st.session_state:
        st.session_state.model = LitWrap().load_from_checkpoint(checkpoint_path, map_location="cpu").eval()

    if file:
        img = Image.open(io.BytesIO(file.read()))
        st.image(img, caption="Input", use_column_width=True)
        x = preprocess(img)
        with torch.inference_mode():
            logits = st.session_state.model(x)
            prob = torch.softmax(logits, dim=1).squeeze().tolist()
            pred = int(torch.argmax(logits, dim=1).item())
        st.subheader(f"Prediction: **{pred}**")
        st.json({str(i): round(p,4) for i,p in enumerate(prob)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    args = parser.parse_args()
    main(args.checkpoint_path)
