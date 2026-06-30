"""Reconstruction engine using trained Pix2Pix GAN."""

from __future__ import annotations
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from app.models.generator import UNetGenerator
from app.utils.checkpoint import load_checkpoint

logger = logging.getLogger("cloudclear_reconstructor")

CHECKPOINT_PATH = Path("checkpoints/pix2pix_liss4_v1_best.pt")


class Pix2PixEngine:
    name = "AI Agent (Pix2Pix GAN)"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from app.config import settings
            import sys
            root_dir = Path(settings.BASE_DIR).parent
            liss_checkpoint = root_dir / "liss4_project" / "checkpoints" / "best_generator.pth"
            
            if liss_checkpoint.exists():
                logger.info("Found trained LISS-IV checkpoint! Loading generator architecture dynamically from liss4_project...")
                liss4_path = str(root_dir / "liss4_project")
                if liss4_path not in sys.path:
                    sys.path.append(liss4_path)
                from liss_models.networks import UNetGenerator as LissUNetGenerator
                
                self.model = LissUNetGenerator(in_channels=3, out_channels=3).to(self.device)
                path = liss_checkpoint
            else:
                logger.info("LISS-IV checkpoint not found in liss4_project. Loading backend U-Net generator...")
                self.model = UNetGenerator(in_channels=3, out_channels=3).to(self.device)
                path = CHECKPOINT_PATH
                if not path.exists():
                    fallback_path = Path("checkpoints/pix2pix_single_inpaint_best.pt")
                    if fallback_path.exists():
                        path = fallback_path
            
            load_checkpoint(path=path, generator=self.model, map_location=self.device)
            self.model.eval()
            logger.info(f"AI model loaded successfully from {path}")
        except Exception as e:
            logger.warning(f"Could not load AI checkpoint: {e}. Using demo mode (OpenCV fallback).")
            self.model = None

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb * 2.0) - 1.0
        chw = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(chw).float().unsqueeze(0)

    def _postprocess(self, pred: torch.Tensor) -> np.ndarray:
        arr = pred.squeeze(0).detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        rgb = (arr * 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def reconstruct(self, input_path: str, mask_path: str, output_path: str) -> bool:
        try:
            image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Cannot read image: {input_path}")

            original_h, original_w = image_bgr.shape[:2]

            # Resize to 256x256 (model input size)
            resized = cv2.resize(image_bgr, (256, 256))

            if self.model is not None:
                tensor = self._preprocess(resized).to(self.device)
                with torch.no_grad():
                    pred = self.model(tensor)
                output = self._postprocess(pred)
            else:
                # Demo fallback: inpaint using mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    output = resized
                else:
                    mask_resized = cv2.resize(mask, (256, 256))
                    output = cv2.inpaint(resized, mask_resized, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Resize back to original dimensions
            output = cv2.resize(output, (original_w, original_h))
            cv2.imwrite(output_path, output)
            logger.info(f"Reconstruction saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return False


# Singleton instance
_engine_instance = None

def get_reconstruction_engine() -> Pix2PixEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = Pix2PixEngine()
    return _engine_instance