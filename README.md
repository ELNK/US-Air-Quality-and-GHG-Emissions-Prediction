# 🏭 US Air Quality and Greenhouse Gas (GHG) Emissions Prediction

This project analyzes US air quality data and GHG emissions data to uncover patterns and build predictive models for air quality across US counties.

## 📁 Project Structure

```
US-Air-Quality-and-GHG-Emissions-Prediction/
├── data/
│   ├── raw/             # Raw input CSVs
│   ├── cleaned/         # Cleaned and preprocessed datasets
│   └── processed/       # Feature-engineered datasets
│
├── figures/             # Visualizations (e.g., heatmaps)
│
├── scripts/
│   ├── __init__.py
│   ├── main.py                # Main orchestration script
│   ├── config.py              # Directory paths and config
│   ├── load_data.py           # Load all raw data
│   ├── clean_data.py          # Clean and rename columns
│   ├── feature_engineering.py # Transform and merge data
│   ├── visualization.py       # Generate charts and maps
│   └── modeling.py            # ML models: linear regression & ANN
```

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure Python 3.7 or newer is installed.  
Then install required Python packages:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install packages manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium
```

---

### ✅ Running the Project

Run from the **project root**, not inside the `scripts/` folder:

```bash
python -m scripts.main
```

This ensures imports like `from scripts.load_data import ...` work correctly.

---

## 🔍 Scripts Overview

### `main.py`
The main entry point. It loads data, cleans and transforms it, creates visuals, and trains models.

### `load_data.py`
Handles loading all raw CSV files from `data/raw/`.

### `clean_data.py`
Standardizes and simplifies column names for easier downstream processing.

### `feature_engineering.py`
Encodes gas types, groups emissions data by county, calculates percentages, and merges datasets.

### `visualization.py`
Creates and saves a correlation heatmap based on engineered features.

### `modeling.py`
Trains and evaluates both linear regression and MLP (neural network) models to predict air quality percentages. Includes residual plots.

---

## 📊 Outputs

- **Cleaned data** → `data/cleaned/`
- **Processed features** → `data/processed/`
- **Charts/plots** → `figures/` (e.g., correlation heatmap)

---

## 🧠 Tips

- Always run scripts from the root project directory (`python -m scripts.main`).
- Don’t run scripts directly from inside the `scripts/` folder unless you adjust import paths.

---

## 📬 Author

- Elina Guo 
- Feel free to reach out with questions or contributions!