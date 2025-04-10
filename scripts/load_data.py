# scripts/load_data.py
import pandas as pd
from config import RAW_DATA_DIR

def load_facilities_data():
    return pd.read_csv(f'{RAW_DATA_DIR}/us_greenhouse_gas_emissions_direct_emitter_facilities.csv')

def load_gas_data():
    return pd.read_csv(f'{RAW_DATA_DIR}/us_greenhouse_gas_emission_direct_emitter_gas_type.csv')

def load_aqi_data():
    return pd.read_csv(f'{RAW_DATA_DIR}/annual_aqi_by_county_2020.csv')

def load_county_mapping():
    return pd.read_csv('data/county_mapping.csv')
