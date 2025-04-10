import os

RAW_DATA_DIR = 'data/raw'
CLEANED_DATA_DIR = 'data/cleaned'
PROCESSED_DATA_DIR = 'data/processed'
FIGURES_DIR = 'figures'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.isfile(path):
        print(f"⚠️ Skipping '{path}': a file with that name already exists.")

# Ensure folders exist safely
ensure_dir(CLEANED_DATA_DIR)
ensure_dir(PROCESSED_DATA_DIR)
ensure_dir(FIGURES_DIR)
