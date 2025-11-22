import os

MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "data/sales_history.csv")
N_RANDOM_SEARCH_ITERATIONS = 60

os.makedirs(MODEL_DIR, exist_ok=True)
data_dir_for_upload = os.path.dirname(DATA_FILE_PATH)
if data_dir_for_upload:
    os.makedirs(data_dir_for_upload, exist_ok=True)

