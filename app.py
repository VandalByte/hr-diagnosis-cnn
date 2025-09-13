import io
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hypertensive Retinopathy Detection Model", page_icon="🩺", layout="centered")

MODELS_DIR = str(Path(__file__).parent / "models")

FILENAME_ARCH_MAP = {
    "efficientnetb0": ("efficientnet_b0", 300),
    "resnet": ("resnet50", 224),
    "densenet": ("densenet121", 224),
    "vgg": ("vgg16_bn", 224),
}

def letterbox_pil(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    im_resized = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im_resized, (x, y))
    return canvas

def make_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def infer_arch_from_filename(p: Path) -> Tuple[str, int]:
    stem = p.stem.lower()
    for key, (arch, sz) in FILENAME_ARCH_MAP.items():
        if key in stem:
            return arch, sz
    return "efficientnet_b0", 300

def safe_load_checkpoint(path: Path) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict.")
    if "model" not in ckpt:
        raise ValueError("Checkpoint missing 'model' state_dict.")
    return ckpt

@st.cache_resource(show_spinner=False)
def load_model_from_path(path_str: str, device: str) -> Tuple[nn.Module, int]:
    p = Path(path_str)
    ckpt = safe_load_checkpoint(p)
    args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    arch = args.get("model")
    img_size = args.get("img_size")
    if arch is None or img_size is None:
        arch, img_size = infer_arch_from_filename(p)
    img_size = int(img_size)
    model = timm.create_model(arch, pretrained=False, num_classes=1, in_chans=3)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, img_size

@torch.inference_mode()
def predict_prob(model: nn.Module, x: torch.Tensor, device: str) -> float:
    logits = model(x.to(device)).squeeze(1)
    return float(torch.sigmoid(logits).item())

class CamHook:
    def __init__(self, module: nn.Module):
        self.activations = None
        self.gradients = None
        self.fwd = module.register_forward_hook(self.forward_hook)
        self.bwd = module.register_full_backward_hook(self.backward_hook)
    def forward_hook(self, _, __, output):
        self.activations = output.detach()
    def backward_hook(self, _, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    def remove(self):
        self.fwd.remove()
        self.bwd.remove()

def find_last_conv_module(model: nn.Module) -> Optional[nn.Module]:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def grad_cam_heatmap(model: nn.Module, x: torch.Tensor, device: str, target_size: int) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    target_layer = find_last_conv_module(model)
    if target_layer is None:
        return np.ones((target_size, target_size), dtype=np.float32) * 0.5
    hook = CamHook(target_layer)
    x = x.to(device).requires_grad_(True)
    out = model(x).squeeze(1)
    if out.ndim == 0:
        out = out.unsqueeze(0)
    score = out
    score.backward(torch.ones_like(score))
    A = hook.activations
    G = hook.gradients
    hook.remove()
    if A is None or G is None:
        return np.ones((target_size, target_size), dtype=np.float32) * 0.5
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1, keepdim=False)
    cam = torch.relu(cam).squeeze(0).detach()
    cam_up = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze()
    cam_np = cam_up.cpu().numpy()
    cam_np -= cam_np.min()
    cam_np /= (cam_np.max() + 1e-8)
    return cam_np.astype(np.float32)

st.sidebar.header("Settings")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: **{device.upper()}**")

def list_checkpoints(folder: str):
    base = Path(folder)
    if not base.exists():
        return []
    files = [str(p.resolve()) for p in base.glob("*.pth")]
    files.sort()
    return files

st.sidebar.button("Rescan")
ckpt_files = list_checkpoints(MODELS_DIR)

model_choice = st.sidebar.selectbox("Select model", options=ckpt_files, index=0 if ckpt_files else None, placeholder="No .pth models found in folder")
threshold = st.sidebar.slider("Decision threshold", 0.00, 1.00, 0.50, 0.01)

st.title("🩺 Hypertensive Retinopathy Detection Model")
st.caption("Upload a fundus image to obtain model probability and label.")

loaded = False
if model_choice:
    try:
        model, img_size = load_model_from_path(model_choice, device)
        loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")

uploaded = st.file_uploader("Choose a fundus image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded and loaded:
    try:
        raw = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()
    disp_im = letterbox_pil(raw, img_size)
    tf = make_transform()
    x = tf(disp_im).unsqueeze(0)
    prob = predict_prob(model, x, device)
    label_text = "**POSITIVE**" if prob >= threshold else "**NEGATIVE**"
    st.subheader("Result")
    st.metric("Probability", f"{prob:.4f}")
    st.write(f"Prediction @ threshold {threshold:.2f}: Hypertensive Retinopathy: {label_text}")
    st.image(disp_im, caption=f"Preprocessed ({img_size}×{img_size})", use_container_width=True)
    st.subheader("Feature Heatmap")
    try:
        heat = grad_cam_heatmap(model, x, device, target_size=img_size)
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(disp_im)
        plt.imshow(heat, alpha=0.4, cmap="jet")
        plt.axis("off")
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Heatmap failed: {e}")
elif not uploaded:
    st.info("Upload an image to run the model.")
