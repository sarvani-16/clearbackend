import os
import cv2
import numpy as np
from pathlib import Path

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
FRONTEND_SAMPLES_DIR = BASE_DIR / "frontend/public/samples"
SAMPLES_DIR.mkdir(exist_ok=True)
FRONTEND_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_1():
    """
    Generates a green countryside tile with fields, a winding river, and cloud blobs.
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [34, 139, 34] # Forest Green
    
    cv2.fillPoly(img, [np.array([[20, 20], [180, 40], [160, 200], [40, 180]])], [80, 150, 200]) # Beige field
    cv2.fillPoly(img, [np.array([[300, 50], [480, 30], [450, 150], [320, 170]])], [50, 100, 150]) # Brown field
    cv2.fillPoly(img, [np.array([[50, 300], [220, 320], [190, 480], [30, 450]])], [70, 120, 90])  # Olive field
    cv2.fillPoly(img, [np.array([[320, 300], [490, 280], [470, 490], [300, 470]])], [100, 180, 140] ) # Light green field

    river_points = np.array([
        [0, 250], [80, 230], [180, 280], [280, 240], [380, 270], [512, 230]
    ], dtype=np.int32)
    for i in range(len(river_points) - 1):
        cv2.line(img, tuple(river_points[i]), tuple(river_points[i+1]), [220, 100, 50], thickness=24) # River
        
    img = cv2.GaussianBlur(img, (5, 5), 0)
    clear_img = img.copy()
    
    cloud_layer = np.zeros_like(img)
    cloud_mask = np.zeros((512, 512), dtype=np.uint8)
    
    cv2.circle(cloud_layer, (120, 130), 65, (255, 255, 255), -1)
    cv2.circle(cloud_mask, (120, 130), 65, 255, -1)
    
    cv2.circle(cloud_layer, (380, 360), 80, (255, 255, 255), -1)
    cv2.circle(cloud_mask, (380, 360), 80, 255, -1)
    
    cv2.circle(cloud_layer, (250, 220), 40, (255, 255, 255), -1)
    cv2.circle(cloud_mask, (250, 220), 40, 255, -1)
    
    cloud_layer = cv2.GaussianBlur(cloud_layer, (41, 41), 0)
    cloud_mask_blurred = cv2.GaussianBlur(cloud_mask, (41, 41), 0)
    
    alpha = cloud_mask_blurred[:, :, np.newaxis] / 255.0
    alpha = alpha * 0.9
    
    cloudy_img = (1 - alpha) * img + alpha * cloud_layer
    cloudy_img = np.clip(cloudy_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(SAMPLES_DIR / "sample_cloudy_1.jpg"), cloudy_img)
    cv2.imwrite(str(FRONTEND_SAMPLES_DIR / "sample_cloudy_1.jpg"), cloudy_img)
    cv2.imwrite(str(SAMPLES_DIR / "sample_clear_1.jpg"), clear_img)
    cv2.imwrite(str(FRONTEND_SAMPLES_DIR / "sample_clear_1.jpg"), clear_img)
    print(f"Generated sample_cloudy_1.jpg and sample_clear_1.jpg in both locations.")

def create_sample_2():
    """
    Generates a coastal view (half blue ocean, half sand/green land) with light clouds.
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [200, 100, 20] # Ocean blue (BGR)
    
    cv2.fillPoly(img, [np.array([[200, 0], [512, 0], [512, 512], [450, 512], [320, 250]])], [150, 220, 240]) # Sand
    cv2.fillPoly(img, [np.array([[350, 0], [512, 0], [512, 512], [500, 512], [420, 250]])], [40, 110, 60]) # Forest
    
    img = cv2.GaussianBlur(img, (9, 9), 0)
    clear_img = img.copy()
    
    cloud_layer = np.zeros_like(img)
    cloud_mask = np.zeros((512, 512), dtype=np.uint8)
    
    cv2.ellipse(cloud_layer, (250, 100), (90, 45), 30, 0, 360, (245, 245, 245), -1)
    cv2.ellipse(cloud_mask, (250, 100), (90, 45), 30, 0, 360, 255, -1)
    
    cv2.circle(cloud_layer, (290, 110), 55, (255, 255, 255), -1)
    cv2.circle(cloud_mask, (290, 110), 55, 255, -1)
    
    cv2.ellipse(cloud_layer, (100, 400), (110, 60), -20, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(cloud_mask, (100, 400), (110, 60), -20, 0, 360, 255, -1)
    
    cloud_layer = cv2.GaussianBlur(cloud_layer, (51, 51), 0)
    cloud_mask_blurred = cv2.GaussianBlur(cloud_mask, (51, 51), 0)
    
    alpha = cloud_mask_blurred[:, :, np.newaxis] / 255.0
    alpha = alpha * 0.85
    
    cloudy_img = (1 - alpha) * img + alpha * cloud_layer
    cloudy_img = np.clip(cloudy_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(SAMPLES_DIR / "sample_cloudy_2.jpg"), cloudy_img)
    cv2.imwrite(str(FRONTEND_SAMPLES_DIR / "sample_cloudy_2.jpg"), cloudy_img)
    cv2.imwrite(str(SAMPLES_DIR / "sample_clear_2.jpg"), clear_img)
    cv2.imwrite(str(FRONTEND_SAMPLES_DIR / "sample_clear_2.jpg"), clear_img)
    print(f"Generated sample_cloudy_2.jpg and sample_clear_2.jpg in both locations.")

def create_sample_default_clear():
    """
    Generates a default clear imagery containing detailed ground features:
    farmlands, forests, gray roads, and building blocks.
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [50, 115, 65] # Default forest green base
    
    # Farmland patches
    cv2.fillPoly(img, [np.array([[10, 10], [240, 10], [240, 240], [10, 240]])], [110, 175, 125]) # Light green crops
    cv2.fillPoly(img, [np.array([[260, 10], [500, 10], [500, 240], [260, 240]])], [70, 140, 190])  # Dry beige/yellow fields
    cv2.fillPoly(img, [np.array([[10, 260], [240, 260], [240, 500], [10, 500]])], [60, 100, 120])  # Brown plowed field
    cv2.fillPoly(img, [np.array([[260, 260], [500, 260], [500, 500], [260, 500]])], [45, 95, 55])   # Dense forest patch

    # Winding gray roads intersecting the land
    cv2.line(img, (250, 0), (250, 512), (180, 180, 180), thickness=8) # Main vertical road
    cv2.line(img, (0, 250), (512, 250), (180, 180, 180), thickness=8) # Main horizontal road
    
    # Add building blocks alongside the roads (beige/gray rectangles)
    # Block 1
    cv2.rectangle(img, (210, 210), (235, 235), (200, 210, 220), -1)
    cv2.rectangle(img, (210, 210), (235, 235), (100, 100, 100), 1)
    # Block 2
    cv2.rectangle(img, (270, 210), (295, 235), (200, 210, 220), -1)
    cv2.rectangle(img, (270, 210), (295, 235), (100, 100, 100), 1)
    # Block 3
    cv2.rectangle(img, (210, 270), (235, 295), (200, 210, 220), -1)
    cv2.rectangle(img, (210, 270), (235, 295), (100, 100, 100), 1)
    # Block 4
    cv2.rectangle(img, (270, 270), (295, 295), (200, 210, 220), -1)
    cv2.rectangle(img, (270, 270), (295, 295), (100, 100, 100), 1)
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Save default clear tile
    cv2.imwrite(str(SAMPLES_DIR / "sample_clear_default.jpg"), img)
    cv2.imwrite(str(FRONTEND_SAMPLES_DIR / "sample_clear_default.jpg"), img)
    print(f"Generated sample_clear_default.jpg in both locations.")

if __name__ == "__main__":
    create_sample_1()
    create_sample_2()
    create_sample_default_clear()
