from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import os
from app.services import get_weather_data, get_satellite_data
from app.utils import calculate_ndvi, predict_fire_risk, create_overlay_image, create_rgb_image

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Data model received from Frontend
class AnalysisRequest(BaseModel):
    bbox: list[float] # [min_lon, min_lat, max_lon, max_lat]
    zoom: float
    start_date: str
    end_date: str

@app.get("/")
def read_root(request: Request):
    # Retrieve the token from environment variables
    token = os.getenv("MAPBOX_TOKEN")
    
    if not token:
        return "Error: MAPBOX_TOKEN missing in the .env file"
    # Inject it into the HTML via Jinja2
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "mapbox_token": token 
    })

@app.post("/api/analyze")
async def analyze_zone(data: AnalysisRequest):
    """
    API: Receives a BBOX, performs Python processing, returns the image in Base64.
    """
    # 1. Zoom check
    if data.zoom < 10:
        return {"status": "ignored", "message": "Zoom too low"}

    bbox = data.bbox
    
    # 2. Call services
    sat_data = get_satellite_data(bbox, data.start_date, data.end_date)
    
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    weather = get_weather_data(bbox[1], bbox[0], bbox[3], bbox[2])
    if sat_data is None:
        return {"status": "error", "message": "No satellite data found."}

    # 3. Calculations
    ndvi = calculate_ndvi(sat_data)
    
    # --- FIX HERE: Removed .values ---
    # ndvi is now a numpy array, mean() returns a numpy scalar directly.
    # We ignore NaNs (Not a Number) to avoid errors if the image has empty pixels
    mean_ndvi = float(ndvi.nanmean()) if hasattr(ndvi, 'nanmean') else float(ndvi.mean())
    
    risk_score = predict_fire_risk(weather, mean_ndvi)
    
    # 4. Generate Images
    img_ndvi_bytes = create_overlay_image(ndvi)
    img_rgb_bytes = create_rgb_image(sat_data)
    
    img_ndvi_b64 = base64.b64encode(img_ndvi_bytes).decode('utf-8')
    img_rgb_b64 = base64.b64encode(img_rgb_bytes).decode('utf-8')

    return {
        "status": "success",
        "risk_score": risk_score,
        "weather": weather,
        "image_ndvi": img_ndvi_b64, 
        "image_rgb": img_rgb_b64,   
        "bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]] 
    }