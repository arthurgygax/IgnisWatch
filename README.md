
# IgnisWatch: Forest Fire Prediction AI

![Python](https://img.shields.io/badge/Backend-FastAPI-009688) ![Frontend](https://img.shields.io/badge/Frontend-Mapbox%20GL-blue) ![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED) ![License](https://img.shields.io/badge/License-MIT-green)

IgnisWatch is an intelligent forest fire monitoring application. It combines real-time satellite imagery (Sentinel-2) and meteorological data to predict fire risks before they spread.

## Project Overview

The goal of this project is to provide a decision-support tool for forest rangers and emergency services. The application allows users to navigate a 3D globe, automatically analyze vegetation health in real-time, and cross-reference this data with local weather conditions to assess fire potential.

### Key Features

* **Fluid Navigation:** GPU-accelerated 3D map rendering using **Mapbox GL JS**.
* **Smart Scanning:** Automatic satellite analysis triggered when the user stops moving the map (Zoom 10+).
* **Satellite Imagery:** Real-time retrieval of Sentinel-2 images via the Microsoft Planetary Computer API.
* **Vegetation Analysis:** Live calculation of NDVI (Normalized Difference Vegetation Index) with smart urban masking (ignoring cities/water).
* **Live Weather:** Integration of temperature, wind, and humidity data (Open-Meteo).
* **Risk Engine:** Algorithm for calculating fire probability (designed to evolve into a Machine Learning model).

## Technical Architecture

The project has evolved into a high-performance Client-Server architecture:

* **Frontend:** HTML5 / JavaScript / Mapbox GL JS
* **Backend:** Python FastAPI (Asynchronous)
* **Data Engineering:** Pandas, Xarray, Rasterio
* **Satellite Provider:** STAC API (Microsoft Planetary Computer)
* **Containerization:** Docker & Docker Compose

## Installation & Setup

### Prerequisites

1. **Docker & Docker Compose** installed.
2. A free **Mapbox Public Access Token** (Get it at [mapbox.com](https://www.mapbox.com)).

### Quick Start

1. **Clone the repository:**

    ```bash
    git clone https://github.com/arthurgygax/IgnisWatch.git
    cd IgnisWatch
    ```

2. **Configure Security (Environment Variables):**
    Create a `.env` file to store your Mapbox API key securely.

    ```bash
    cp .env.example .env

    ```

    Open the `.env` file and paste your key:

    ```env
    MAPBOX_TOKEN=pk.eyJ1Ijo...your_key_here...
    ```

3. **Launch the application:**
    This command builds the container and starts the FastAPI server.

    ```bash
    docker-compose up --build
    ```

4. **Access the Dashboard:**
    Open your browser to: `http://localhost:8000`

## Roadmap

The project is divided into two major phases:

**Phase 1: Real-time Architecture** (Current Status)

- [x] Implementation of the Client-Server architecture (FastAPI).
- [x] Integration of Mapbox GL for fluid rendering.
- [x] Automatic "Move-and-Scan" trigger logic.
- [x] Implementation of the heuristic risk detection algorithm.

**Phase 2: Artificial Intelligence** (Upcoming)

- [ ] Collection of historical fire data (NASA FIRMS).
- [ ] Training a Machine Learning model (XGBoost/Random Forest) on past events.
- [ ] Replacement of the heuristic algorithm with the predictive model.

## Project Structure

```bash
IgnisWatch/
├── app/
│   ├── main.py            # FastAPI Server & Routes
│   ├── templates/         # Frontend (HTML/JS Mapbox)
│   │   └── index.html
│   ├── services.py        # API Connectors (Satellite & Weather)
│   └── utils.py           # Calculation Logic (NDVI, Image Gen)
├── .env                   # API Keys (Not committed to Git)
├── Dockerfile             # Python Environment + GDAL
├── docker-compose.yml     # Orchestration
└── requirements.txt       # Python Dependencies

```

## Contributing

Contributions are welcome. Please feel free to open an Issue or submit a Pull Request.

## License

Distributed under the MIT License. See LICENSE for more information.
