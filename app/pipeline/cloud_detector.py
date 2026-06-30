import os
import cv2
import numpy as np
from PIL import Image as PILImage
from app.config import settings

# Defensive check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = object  # dummy base class for UNet definition to avoid SyntaxError

# --- PyTorch U-Net Architecture Definition (Only defined if torch is available) ---
if HAS_TORCH:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            self.inc = DoubleConv(in_channels, 64)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
            
            self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.conv_up1 = DoubleConv(512, 256)
            
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.conv_up2 = DoubleConv(256, 128)
            
            self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.conv_up3 = DoubleConv(128, 64)
            
            self.outc = nn.Conv2d(64, out_channels, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            
            x = self.up1(x4)
            x = torch.cat([x, x3], dim=1)
            x = self.conv_up1(x)
            
            x = self.up2(x)
            x = torch.cat([x, x2], dim=1)
            x = self.conv_up2(x)
            
            x = self.up3(x)
            x = torch.cat([x, x1], dim=1)
            x = self.conv_up3(x)
            
            logits = self.outc(x)
            return torch.sigmoid(logits)
else:
    UNet = None


class CloudDetector:
    def __init__(self):
        self.use_pytorch = False
        self.model = None
        self.device = None
        
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                       ("mps" if torch.backends.mps.is_available() else "cpu"))
            self.model_path = os.path.join(settings.MODELS_FOLDER, "u_net.pt")
            
            # Try loading PyTorch U-Net
            if os.path.exists(self.model_path):
                try:
                    print(f"[CloudDetector] Loading U-Net weights from: {self.model_path}")
                    self.model = UNet(in_channels=3, out_channels=1)
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    self.model.to(self.device)
                    self.model.eval()
                    self.use_pytorch = True
                    print("[CloudDetector] PyTorch U-Net initialized successfully.")
                except Exception as e:
                    print(f"[CloudDetector] Failed to load PyTorch model: {e}. Falling back to CV.")
            else:
                print("[CloudDetector] U-Net weight file 'u_net.pt' not found in models/. Using OpenCV fallback.")
        else:
            print("[CloudDetector] PyTorch is not installed in the current environment. Using OpenCV fallback.")

    def detect(self, image_path: str, mask_output_path: str) -> float:
        """
        Detects clouds and saves the binary cloud mask.
        Returns the cloud coverage percentage.
        """
        if self.use_pytorch and self.model is not None and HAS_TORCH:
            return self._detect_pytorch(image_path, mask_output_path)
        else:
            return self._detect_cv(image_path, mask_output_path)

    def _detect_pytorch(self, image_path: str, mask_output_path: str) -> float:
        try:
            # Load and preprocess image
            img = PILImage.open(image_path).convert("RGB")
            orig_w, orig_h = img.size
            
            # Resize to standard model size (512x512)
            img_resized = img.resize((512, 512))
            img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device) # [1, 3, 512, 512]
            
            with torch.no_grad():
                mask_pred = self.model(img_tensor) # [1, 1, 512, 512]
                mask_pred = mask_pred.squeeze().cpu().numpy()
            
            # Threshold predictions to create a binary mask
            binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255
            
            # Resize mask back to original image size
            binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Save mask
            cv2.imwrite(mask_output_path, binary_mask)
            
            # Calculate percentage
            cloud_pixels = np.sum(binary_mask == 255)
            total_pixels = binary_mask.size
            cloud_percentage = (cloud_pixels / total_pixels) * 100.0
            
            return float(round(cloud_percentage, 2))
        except Exception as e:
            print(f"[CloudDetector] PyTorch inference error: {e}. Trying CV fallback.")
            return self._detect_cv(image_path, mask_output_path)

    def _detect_cv(self, image_path: str, mask_output_path: str) -> float:
        """
        Advanced CV based segmentation:
        1. Reads image.
        2. Converts to HSL space (Hue, Saturation, Lightness).
        3. Thresholds Lightness >= 185 (bright regions) and Saturation <= 75 (white/grey clouds, not saturated ground).
        4. Applies morphological opening to remove small sensor noise, and closing to clean boundaries.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        h, w, _ = img.shape
        
        # Convert to HLS (Hue, Lightness, Saturation)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        _, l_channel, s_channel = cv2.split(hls)
        
        # Brightness threshold (high L) and saturation threshold (low S)
        bright_mask = cv2.threshold(l_channel, 185, 255, cv2.THRESH_BINARY)[1]
        desaturated_mask = cv2.threshold(s_channel, 75, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Combine masks
        combined = cv2.bitwise_and(bright_mask, desaturated_mask)
        
        # Apply morphological operations to clean up
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        cleaned_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Save mask image
        cv2.imwrite(mask_output_path, cleaned_mask)
        
        # Calculate percentage
        cloud_pixels = np.sum(cleaned_mask == 255)
        total_pixels = cleaned_mask.size
        cloud_percentage = (cloud_pixels / total_pixels) * 100.0
        
        return float(round(cloud_percentage, 2))
