import numpy as np
import io
from PIL import Image

def calculate_ndvi(dataset):
    """Computes NDVI (NIR - RED) / (NIR + RED)"""
    red = dataset["B04"].values.astype("float32")
    nir = dataset["B08"].values.astype("float32")
    
    denominator = nir + red
    denominator[denominator == 0] = np.nan
    
    ndvi = (nir - red) / denominator
    return ndvi

def predict_fire_risk(weather, mean_ndvi):
    """Calculates Risk Score (0-100)."""
    score = 0
    if weather['temp'] > 25: score += 20
    if weather['temp'] > 35: score += 20
    if weather['humidity'] < 40: score += 20
    if weather['humidity'] < 20: score += 20
    if weather['wind'] > 15: score += 10
    if mean_ndvi < 0.3: score += 30 
    elif mean_ndvi < 0.5: score += 10
    return min(score, 100)

def create_overlay_image(ndvi_data):
    """Generates the Risk/NDVI Map."""
    # 1. Handle NaNs
    ndvi_clean = np.nan_to_num(ndvi_data, nan=-1.0)

    # 2. Normalize
    norm_data = (ndvi_clean + 1) / 2
    norm_data = np.clip(norm_data, 0, 1)
    img_array = (norm_data * 255).astype(np.uint8)

    # 3. Apply Colors
    h, w = img_array.shape
    rgba_img = np.zeros((h, w, 4), dtype=np.uint8)

    mask_red = img_array < 128
    rgba_img[mask_red] = [215, 48, 39, 160] 

    mask_yellow = (img_array >= 128) & (img_array < 192)
    rgba_img[mask_yellow] = [253, 231, 37, 160] 

    mask_green = img_array >= 192
    rgba_img[mask_green] = [26, 152, 80, 160] 

    # 4. Save
    image = Image.fromarray(rgba_img, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()

def create_rgb_image(dataset):
    """
    Generates a True Color (RGB) Image from satellite data.
    """
    # 1. Extract bands
    r = dataset["B04"].values.astype("float32")
    g = dataset["B03"].values.astype("float32")
    b = dataset["B02"].values.astype("float32")

    # 2. Stack into an RGB image
    rgb = np.dstack((r, g, b))

    # 3. Handle Empty Data (NaNs) -> Make transparent
    # We create an Alpha channel where data exists
    alpha = np.where(np.isnan(r), 0, 255).astype(np.uint8)
    
    # Replace NaNs with 0 to avoid errors
    rgb = np.nan_to_num(rgb, nan=0.0)

    # 4. Brightness Normalization
    # Sentinel-2 raw values usually go up to ~3000 for land.
    # We clip at 3000 to avoid "washed out" white images.
    rgb = np.clip(rgb, 0, 3000) / 3000.0
    
    # Convert to 0-255
    rgb = (rgb * 255).astype(np.uint8)

    # 5. Add Alpha Channel
    rgba = np.dstack((rgb, alpha))

    # 6. Save
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()