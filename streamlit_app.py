"""Streamlit app for Pix2Pix cloud-removal inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from config import Config
from models.generator import UNetGenerator
from utils.checkpoint import load_checkpoint


@st.cache_resource
def load_model(checkpoint_path: str, device_str: str) -> tuple[UNetGenerator, torch.device]:
    cfg = Config()
    device = torch.device(device_str)
    model = UNetGenerator(in_channels=cfg.data.num_channels, out_channels=cfg.data.num_channels).to(device)
    load_checkpoint(path=Path(checkpoint_path), generator=model, map_location=device)
    model.eval()
    return model, device


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb * 2.0) - 1.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).float().unsqueeze(0)


def postprocess(pred: torch.Tensor) -> np.ndarray:
    arr = pred.squeeze(0).detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def main() -> None:
    st.set_page_config(page_title="LISS-IV Cloud Removal", layout="wide")
    st.title("Generative AI Cloud Removal (Pix2Pix)")
    checkpoint = st.text_input("Checkpoint path", value="checkpoints/pix2pix_liss4_cloud_removal_best.pt")
    device_option = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    uploaded = st.file_uploader("Upload cloudy image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not decode image.")
        return

    model, device = load_model(checkpoint, device_option)
    with torch.no_grad():
        pred = model(preprocess(image_bgr).to(device))
    pred_rgb = postprocess(pred)
    in_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    col1.image(in_rgb, caption="Cloudy Input", use_container_width=True)
    col2.image(pred_rgb, caption="Generated Clear Output", use_container_width=True)


if __name__ == "__main__":
    main()

