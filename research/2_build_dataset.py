import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
import pystac_client
import planetary_computer
import odc.stac
from tqdm import tqdm

# --- CONFIG ---
INPUT_FILE = "research/sampled_points_to_process.csv"
OUTPUT_FILE = "research/final_training_dataset.csv"
FETCH_SATELLITE = True 

# --- API SETUP ---
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def get_historical_weather(lat, lon, date_str):
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "daily": ["temperature_2m_max", "wind_speed_10m_max", "precipitation_sum"],
            "hourly": ["relative_humidity_2m"],
            "timezone": "auto"
        }
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200: return None, None, None, None
        data = r.json()
        
        temp_max = data['daily']['temperature_2m_max'][0]
        wind_max = data['daily']['wind_speed_10m_max'][0]
        rain_sum = data['daily']['precipitation_sum'][0]
        humidity = data['hourly']['relative_humidity_2m'][12] # Noon
        return temp_max, humidity, wind_max, rain_sum
    except:
        return None, None, None, None

def get_historical_ndvi(lat, lon, date_str):
    if not FETCH_SATELLITE: return 0.0
    try:
        box_size = 0.02
        bbox = [lon - box_size, lat - box_size, lon + box_size, lat + box_size]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_search = (date_obj - timedelta(days=20)).strftime("%Y-%m-%d") # Look back 20 days
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_search}/{date_str}",
            query={"eo:cloud_cover": {"lt": 25}},
            limit=1
        )
        items = search.item_collection()
        if not items: return None
        
        data = odc.stac.load(
            items, bands=["B04", "B08"], bbox=bbox,
            resolution=100, chunks={}
        ).median(dim="time").compute()
        
        red = data["B04"].values.astype("float32")
        nir = data["B08"].values.astype("float32")
        ndvi = (nir - red) / (nir + red + 1e-6)
        return float(np.nanmean(ndvi))
    except:
        return None

# --- PROCESS ---
if not os.path.exists(INPUT_FILE):
    print("Input file not found. Run Step 1 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"Starting enrichment for {len(df)} samples...")

# Check if we are resuming
start_idx = 0
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    start_idx = len(existing_df)
    print(f"Resuming from index {start_idx}...")
else:
    # Create header
    pd.DataFrame(columns=['latitude','longitude','date','temperature','humidity','wind_speed','rainfall','ndvi','fire_occurred']).to_csv(OUTPUT_FILE, index=False)

# Loop
for index, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df)-start_idx):
    lat, lon, date, target = row['latitude'], row['longitude'], row['acq_date'], row['fire_occurred']
    
    # 1. Fetch
    temp, hum, wind, rain = get_historical_weather(lat, lon, date)
    
    ndvi = None
    if temp is not None:
        ndvi = get_historical_ndvi(lat, lon, date)
    
    # 2. Append if valid
    if temp is not None and ndvi is not None:
        new_row = pd.DataFrame([{
            'latitude': lat, 'longitude': lon, 'date': date,
            'temperature': temp, 'humidity': hum, 'wind_speed': wind, 'rainfall': rain,
            'ndvi': ndvi, 'fire_occurred': target
        }])
        
        # Save incrementally (Append mode)
        new_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    
    time.sleep(0.5) # Rate limit protection

print("Data Collection Complete!")