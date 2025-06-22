import streamlit as st
import torch
import torch.nn as nn

# Hyperparameters matching training
nz = 100
n_classes = 10

# Generator definition must match training
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, nz)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, 64 * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        e = self.label_emb(labels)
        x = torch.cat([z, e], dim=1).view(-1, nz * 2, 1, 1)
        return self.net(x)

@st.cache_resource
def load_generator():
    gen = Generator().cpu()
    state = torch.load('generator.pth', map_location='cpu')
    gen.load_state_dict(state)
    gen.eval()
    return gen

# Streamlit UI
st.title("Handwritten Digit Generator")
st.write("Generate synthetic MNIST-like handwritten digits.")

digit = st.selectbox("Select digit (0-9)", range(n_classes))
if st.button("Generate 5 Images"):
    gen = load_generator()
    noise = torch.randn(5, nz)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = gen(noise, labels)
    imgs = (imgs + 1) / 2.0  # normalize to [0,1]

    cols = st.columns(5)
    for i, col in enumerate(cols):
        img = imgs[i].squeeze().numpy()
        col.image(img, width=80)
        col.caption(f"Sample {i+1}")
