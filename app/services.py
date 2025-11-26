import pystac_client
import planetary_computer
import odc.stac
import openmeteo_requests
import requests_cache
from retry_requests import retry
from functools import lru_cache
import numpy as np

# --- WEATHER API ---
@lru_cache(maxsize=100) 
def get_weather_data(lat_min, lon_min, lat_max, lon_max):
    """
    Fetches weather for a 4x4 Grid (16 points) across the view.
    Adaptative: The grid scales with the BBOX (Zoom).
    """
    # 1. Generate a 4x4 grid of coordinates
    # linspace generates evenly spaced numbers over a specified interval
    lat_steps = np.linspace(lat_max, lat_min, 4) # Top to Bottom
    lon_steps = np.linspace(lon_min, lon_max, 4) # Left to Right
    
    lats = []
    lons = []
    
    # Create the meshgrid
    for lat in lat_steps:
        for lon in lon_steps:
            lats.append(lat)
            lons.append(lon)
            
    # 2. Call Open-Meteo
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lats,
        "longitude": lons,
        "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        
        grid_data = []
        avg_temp = 0
        avg_wind = 0
        avg_hum = 0
        avg_wind_dir_x = 0
        avg_wind_dir_y = 0
        
        for i, response in enumerate(responses):
            current = response.Current()
            temp = current.Variables(0).Value()
            hum = current.Variables(1).Value()
            wind_speed = current.Variables(2).Value()
            wind_dir = current.Variables(3).Value()
            
            # Statistics
            avg_temp += temp
            avg_hum += hum
            avg_wind += wind_speed
            
            # Vector math for average direction
            rad = np.radians(wind_dir)
            avg_wind_dir_x += np.sin(rad)
            avg_wind_dir_y += np.cos(rad)
            
            grid_data.append({
                "u": wind_speed,
                "v": wind_dir, # Degrees
                "lat": lats[i],
                "lon": lons[i]
            })

        count = len(responses)
        
        # Calculate mean wind direction in degrees
        avg_dir_deg = np.degrees(np.arctan2(avg_wind_dir_x, avg_wind_dir_y))
        if avg_dir_deg < 0: avg_dir_deg += 360

        return {
            "temp": avg_temp / count,
            "humidity": avg_hum / count,
            "wind": avg_wind / count,
            "wind_dir": avg_dir_deg,
            "grid": grid_data # 16 points
        }

    except Exception as e:
        print(f"Weather Grid Error: {e}")
        return None

# --- SATELLITE API ---
# We use a tuple for bbox to make it hashable for lru_cache
@lru_cache(maxsize=32)
def fetch_satellite_data_cached(bbox_tuple, start_date, end_date):
    """
    Internal function to cache the heavy lifting.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox_tuple,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 25}},
    )
    
    items = search.item_collection()
    if not items:
        return None

    # UPDATED: We now fetch B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
    data = odc.stac.load(
        items,
        bands=["B02", "B03", "B04", "B08"], 
        bbox=bbox_tuple,
        resolution=0.0002, 
        crs="EPSG:4326",
        chunks={} 
    )
    
    if 'time' in data.dims:
        # The .median() operation effectively constructs the best possible image
        # over the selected period (removing clouds and moving objects).
        # This represents the "Actual Map" during the selected timeframe.
        data = data.median(dim="time", keep_attrs=True).compute()
        
    return data

def get_satellite_data(bbox, start_date, end_date):
    rounded_bbox = tuple([round(x, 3) for x in bbox])
    return fetch_satellite_data_cached(rounded_bbox, start_date, end_date)