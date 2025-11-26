    
# IgnisWatch: Forest Fire Prediction AI

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED) ![License](https://img.shields.io/badge/License-MIT-green)

IgnisWatch is an intelligent forest fire monitoring application. It combines real-time satellite imagery (Sentinel-2) and meteorological data to predict fire risks before they spread.

## Project Overview

The goal of this project is to provide a decision-support tool for forest rangers and emergency services. The application allows users to select a specific region anywhere on the globe, analyze vegetation health, and cross-reference this data with local weather conditions to assess fire potential.

### Key Features
*   **Interactive Interface:** Dynamic map to draw and select monitoring zones (Leaflet/Folium).
*   **Satellite Imagery:** Automatic retrieval of Sentinel-2 images via the Microsoft Planetary Computer API.
*   **Vegetation Analysis:** Real-time calculation of NDVI (Normalized Difference Vegetation Index) to measure moisture and plant health.
*   **Live Weather:** Integration of temperature, wind, and humidity data (Open-Meteo).
*   **Risk Engine:** Algorithm for calculating fire probability (designed to evolve into a Machine Learning model).

## Technical Architecture

The project is designed to be modular, containerized, and easily deployable.

*   **Frontend:** Streamlit (Python)
*   **Mapping:** Folium & Leaflet
*   **Data Engineering:** Pandas, Xarray, Rasterio
*   **Satellite Provider:** STAC API (Microsoft Planetary Computer)
*   **Containerization:** Docker & Docker Compose

## Installation & Setup

### Prerequisites
*   Docker & Docker Compose installed on your machine.

### Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arthurgygax/IgnisWatch.git
    cd IgnisWatch
    ```

2.  **Launch the application (via Docker):**
    This command builds the image (installing GDAL and geospatial libraries) and starts the server.
    ```bash
    docker-compose up --build
    ```

3.  **Access the Dashboard:**
    Open your browser to: `http://localhost:8501`

## Roadmap

The project is divided into two major phases:

**Phase 1: Application & Data Pipeline** (Current Status)
- [x] Implementation of the mapping interface.
- [x] Connection to free Satellite and Weather APIs.
- [x] Implementation of the heuristic risk detection algorithm.

**Phase 2: Artificial Intelligence** (Upcoming)
- [ ] Collection of historical fire data (NASA FIRMS).
- [ ] Training a Machine Learning model (XGBoost/Random Forest) on past events.
- [ ] Replacement of the heuristic algorithm with the predictive model.

## Project Structure

```bash
IgnisWatch/
├── app/
│   ├── main.py            # Streamlit Interface
│   ├── services.py        # API Connectors (Satellite & Weather)
│   └── utils.py           # Calculation Logic (NDVI, Risk Score)
├── Dockerfile             # Python Environment + GDAL
├── docker-compose.yml     # Orchestration
└── requirements.txt       # Python Dependencies
```
## Contributing

Contributions are welcome. Please feel free to open an Issue or submit a Pull Request.

## License

Distributed under the MIT License. See LICENSE for more information.
