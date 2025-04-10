# scripts/main.py
from load_data import load_facilities_data, load_gas_data, load_aqi_data, load_county_mapping
from clean_data import clean_facility_columns, clean_gas_columns
from feature_engineering import one_hot_encode_gas, aggregate_by_county, compute_aqi_percent
from visualization import correlation_heatmap
from config import PROCESSED_DATA_DIR
import pandas as pd

# === Load and Clean ===
facilities = clean_facility_columns(load_facilities_data())
gas = clean_gas_columns(load_gas_data())
aqi = load_aqi_data()
county_map = load_county_mapping()

# === Feature Engineering ===
gas_encoded = one_hot_encode_gas(gas)
gas_encoded.to_csv(f"{PROCESSED_DATA_DIR}/gas_encoded.csv", index=False)

# Further steps: aggregation, merging, modeling...

# Example: Save a cleaned file
aqi_cleaned = compute_aqi_percent(aqi)

aqi_cleaned.to_csv('data/cleaned/aqi_cleaned.csv', index=False)
