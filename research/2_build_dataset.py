import pandas as pd
import numpy as np
import requests
import time
import os
import warnings
from datetime import datetime, timedelta
import pystac_client
import planetary_computer
import odc.stac
from tqdm import tqdm
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
INPUT_FILE = "research/sampled_points_to_process.csv"
OUTPUT_FILE = "research/final_advanced_dataset.csv"
IMAGE_FOLDER = "research/images/"
FETCH_SATELLITE = True 

# Create image folder
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --- API SETUP ---
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# --- IMAGE GENERATION LOGIC (MATCHING APP) ---

def save_rgb_image_to_disk(r, g, b, filename):
    """
    Generates True Color RGB using exact App logic:
    - Clip at 3000 (Brightness)
    - NaN handling -> Transparent
    """
    try:
        # 1. Stack
        rgb = np.dstack((r, g, b))

        # 2. Handle Empty Data (NaNs) -> Make transparent
        # We create an Alpha channel where data exists
        # Check if R band has data
        alpha = np.where(np.isnan(r), 0, 255).astype(np.uint8)
        
        # Replace NaNs with 0 to avoid errors in math
        rgb = np.nan_to_num(rgb, nan=0.0)

        # 3. Brightness Normalization (The App Logic)
        # Sentinel-2 raw values usually go up to ~3000 for land.
        rgb = np.clip(rgb, 0, 3000) / 3000.0
        
        # Convert to 0-255
        rgb = (rgb * 255).astype(np.uint8)

        # 4. Add Alpha Channel
        rgba = np.dstack((rgb, alpha))

        # 5. Save
        img = Image.fromarray(rgba, mode="RGBA")
        path = os.path.join(IMAGE_FOLDER, filename)
        img.save(path, format="PNG", optimize=True)
        return filename
    except Exception:
        return None

def save_ndvi_image_to_disk(ndvi_map, filename):
    """
    Generates NDVI Map using exact App logic:
    - Normalization -1..1 to 0..255
    - Thresholds: <128 (Red), 128-192 (Yellow), >192 (Green)
    """
    try:
        # 1. Handle NaNs
        ndvi_clean = np.nan_to_num(ndvi_map, nan=-1.0)

        # 2. Normalize (-1 to 1 -> 0 to 255)
        norm_data = (ndvi_clean + 1) / 2
        norm_data = np.clip(norm_data, 0, 1)
        img_array = (norm_data * 255).astype(np.uint8)

        # 3. Apply Colors (Exact App Logic)
        h, w = img_array.shape
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)

        # RED (Low NDVI / Water / Concrete) < 128
        mask_red = img_array < 128
        # Note: Used 255 alpha (Solid) for dataset readability instead of 160
        rgba_img[mask_red] = [215, 48, 39, 255] 

        # YELLOW (Mid NDVI) 128 to 192
        mask_yellow = (img_array >= 128) & (img_array < 192)
        rgba_img[mask_yellow] = [253, 231, 37, 255] 

        # GREEN (High NDVI) > 192
        mask_green = img_array >= 192
        rgba_img[mask_green] = [26, 152, 80, 255] 

        # Handle explicit NaNs/Empty as transparent
        # If the original input was NaN, ndvi_clean became -1.0 -> img_array 0.
        # So pixels with value 0 (Water/Dead/Empty) are Red.
        # If you want empty space to be transparent, we check original NaNs:
        rgba_img[np.isnan(ndvi_map)] = [0, 0, 0, 0]

        # 4. Save
        img = Image.fromarray(rgba_img, mode="RGBA")
        path = os.path.join(IMAGE_FOLDER, filename)
        img.save(path, format="PNG", optimize=True)
        return filename
    except Exception:
        return None

# --- DATA FETCHING ---

def get_historical_weather(lat, lon, date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (date_obj - timedelta(days=30)).strftime("%Y-%m-%d")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": date_str,
            "daily": ["temperature_2m_max", "precipitation_sum"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
            "timezone": "auto"
        }
        
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200: return None
        data = r.json()
        
        # Noon index
        temp_live = data['hourly']['temperature_2m'][-12]
        hum_live = data['hourly']['relative_humidity_2m'][-12]
        wind_live = data['hourly']['wind_speed_10m'][-12]
        
        temps = data['daily']['temperature_2m_max']
        rains = data['daily']['precipitation_sum']
        
        return {
            "temperature": temp_live,
            "humidity": hum_live,
            "wind_speed": wind_live,
            "rain_7d_sum": sum(rains[-7:]),
            "rain_30d_sum": sum(rains),
            "temp_7d_avg": sum(temps[-7:]) / 7,
            "temp_30d_max": max(temps)
        }
    except:
        return None

def get_satellite_data_raw(lat, lon, date_str):
    """
    Returns raw bands (numpy arrays) instead of processed images.
    """
    if not FETCH_SATELLITE: return None, None, None, None
    try:
        box_size = 0.03
        bbox = [lon - box_size, lat - box_size, lon + box_size, lat + box_size]
        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Look back 20 days
        start_search = (date_obj - timedelta(days=20)).strftime("%Y-%m-%d")
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_search}/{date_str}",
            query={"eo:cloud_cover": {"lt": 30}}, 
            limit=5
        )
        items = search.item_collection()
        if not items: return None, None, None, None
        
        # Load Raw Bands
        ds = odc.stac.load(
            items, bands=["B02", "B03", "B04", "B08"], bbox=bbox,
            resolution=20, chunks={}
        )
        
        # Median over time
        combined = ds.median(dim="time").compute()
        
        red = combined["B04"].values.astype("float32")
        nir = combined["B08"].values.astype("float32")
        green = combined["B03"].values.astype("float32")
        blue = combined["B02"].values.astype("float32")
        
        return red, nir, green, blue
        
    except Exception as e:
        return None, None, None, None

# --- MAIN LOOP ---
if not os.path.exists(INPUT_FILE):
    print("❌ Input file not found.")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"🚀 Starting Data Collection (Exact App Logic)...")

start_idx = 0
if os.path.exists(OUTPUT_FILE):
    start_idx = len(pd.read_csv(OUTPUT_FILE))
    print(f"🔄 Resuming from index {start_idx}...")
else:
    cols = ['latitude','longitude','date','fire_occurred',
            'temperature','humidity','wind_speed',
            'rain_7d_sum','rain_30d_sum','temp_7d_avg','temp_30d_max',
            'ndvi','rgb_image','ndvi_image']
    pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)

for index, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df)-start_idx):
    lat, lon, date, target = row['latitude'], row['longitude'], row['acq_date'], row['fire_occurred']
    
    # 1. Weather
    w = get_historical_weather(lat, lon, date)
    
    # 2. Satellite
    if w:
        r, nir, g, b = get_satellite_data_raw(lat, lon, date)
        
        if r is not None:
            # Calc NDVI Scalar for CSV
            denominator = nir + r
            denominator[denominator == 0] = np.nan
            ndvi_map = (nir - r) / denominator
            ndvi_val = float(np.nanmean(ndvi_map))
            
            # Save Images using App Logic
            base_name = f"{date}_{lat}_{lon}".replace(".", "")
            
            rgb_filename = f"{base_name}_rgb.png"
            ndvi_filename = f"{base_name}_ndvi.png"
            
            saved_rgb = save_rgb_image_to_disk(r, g, b, rgb_filename)
            saved_ndvi = save_ndvi_image_to_disk(ndvi_map, ndvi_filename)
            
            if saved_rgb and saved_ndvi:
                new_row = pd.DataFrame([{
                    'latitude': lat, 'longitude': lon, 'date': date, 'fire_occurred': target,
                    'temperature': w['temperature'], 'humidity': w['humidity'], 'wind_speed': w['wind_speed'],
                    'rain_7d_sum': w['rain_7d_sum'], 'rain_30d_sum': w['rain_30d_sum'],
                    'temp_7d_avg': w['temp_7d_avg'], 'temp_30d_max': w['temp_30d_max'],
                    'ndvi': ndvi_val, 
                    'rgb_image': saved_rgb, 
                    'ndvi_image': saved_ndvi
                }])
                
                new_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    
    time.sleep(0.2)

print("✅ Collection Complete!")