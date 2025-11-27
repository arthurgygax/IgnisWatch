import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Path to the folder containing the country CSVs
INPUT_FOLDER = "research/nasa_data/2023/" 
OUTPUT_FILE = "research/sampled_points_to_process.csv"

# Target: How many samples do we want for the training?
# 5,000 Fires + 5,000 Non-Fires = 10,000 rows.
# This fits within the free tiers of APIs and computes in ~6-8 hours.
TARGET_FIRE_COUNT = 50

def shift_date(date_str):
    """
    Shifts date by ~6 months (180 days) to create a 'Safe' example.
    Rationale: If a place burns in Summer, it's usually safe in Winter.
    """
    try:
        # VIIRS often uses YYYY-MM-DD
        d = datetime.strptime(str(date_str), "%Y-%m-%d")
        new_d = d - timedelta(days=180)
        return new_d.strftime("%Y-%m-%d")
    except:
        return None

# 1. FIND ALL FILES
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
print(f"Found {len(csv_files)} country files.")

all_fires = []

# 2. ITERATE AND FILTER
print("Scanning for high-confidence fires...")

# We shuffle the file list so we don't just get countries starting with 'A'
np.random.shuffle(csv_files)

for file in csv_files:
    try:
        df = pd.read_csv(file)
        
        # Standardize column names (Lowercase)
        df.columns = [c.lower() for c in df.columns]
        
        # Filter High Confidence
        # VIIRS 'confidence' is usually 'l' (low), 'n' (nominal), 'h' (high).
        # We accept 'n' and 'h' to get enough data, but strictly 'h' is better if enough data.
        if 'confidence' in df.columns:
            # Check if column is string (l, n, h) or number
            if df['confidence'].dtype == 'O': 
                high_conf = df[df['confidence'].isin(['h', 'high'])] # Strict High
            else:
                high_conf = df[df['confidence'] >= 80] # Numeric equivalent
        else:
            # Fallback if no confidence column (rare)
            high_conf = df
            
        # Optimization: Don't take ALL fires from one country. 
        # Take max 50 per country to ensure diversity.
        if len(high_conf) > 0:
            sample_size = min(len(high_conf), 50)
            all_fires.append(high_conf.sample(n=sample_size))
            
    except Exception as e:
        print(f"Skipped {os.path.basename(file)}: {e}")

    # Stop if we have enough candidates (x2 buffer to be safe)
    current_count = sum([len(x) for x in all_fires])
    if current_count > TARGET_FIRE_COUNT * 1.5:
        break

# 3. CONSOLIDATE
if not all_fires:
    print("No fires found. Check your folder path.")
    exit()

df_fires = pd.concat(all_fires)
print(f"Collected {len(df_fires)} high-confidence fires candidates.")

# Final random sample to hit exact target
if len(df_fires) > TARGET_FIRE_COUNT:
    df_fires = df_fires.sample(n=TARGET_FIRE_COUNT, random_state=42)

# 4. PREPARE DATASET (Fire vs Non-Fire)
# Positive Class (1)
fires_final = df_fires[['latitude', 'longitude', 'acq_date']].copy()
fires_final['fire_occurred'] = 1

# Negative Class (0) - Time Shifted
non_fires = fires_final.copy()
non_fires['acq_date'] = non_fires['acq_date'].apply(shift_date)
non_fires['fire_occurred'] = 0

# Merge and Shuffle
final_df = pd.concat([fires_final, non_fires]).sample(frac=1).reset_index(drop=True)

# 5. SAVE
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"SUCCESS: Saved {len(final_df)} samples to '{OUTPUT_FILE}'")
print("You can now run '2_build_dataset.py' to enrich this data with Weather & Satellites.")