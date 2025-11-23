import pandas as pd
import os
import logging
from app.config import DATA_FILE_PATH

logger = logging.getLogger(__name__)

def get_product_data(sku: str) -> pd.DataFrame | None:
    try:
        logger.info(f"Attempting to read data file: {DATA_FILE_PATH}")
        if not os.path.exists(DATA_FILE_PATH):
            logger.error(f"Data file not found at {DATA_FILE_PATH}")
            return None
        df_all = pd.read_csv(DATA_FILE_PATH, names=['sku', 'qty_ordered', 'created_at'], header=None)
        df_all['sku'] = df_all['sku'].astype(str).str.strip().str.lower()
        logger.info(f"Successfully read data file. Total rows: {len(df_all)}")
    except pd.errors.EmptyDataError:
        logger.warning(f"Data file {DATA_FILE_PATH} is empty.")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV file {DATA_FILE_PATH}: {e}", exc_info=True)
        return None

    sku_cleaned = str(sku).strip().lower()
    df_product_raw = df_all[df_all['sku'] == sku_cleaned].copy()

    if df_product_raw.empty:
        logger.warning(f"No data found for SKU: {sku} in the CSV.")
        return None

    try:
        df_product_raw['created_at'] = pd.to_datetime(df_product_raw['created_at']).dt.tz_localize(None)
        df_daily = df_product_raw.set_index('created_at').resample('D')['qty_ordered'].sum().reset_index()
        df_daily.rename(columns={"created_at": "ds", "qty_ordered": "y"}, inplace=True)
        df_daily = df_daily.sort_values('ds')
        logger.info(f"Aggregated data for SKU {sku}. Found {len(df_daily)} daily records.")
    except Exception as e:
        logger.error(f"Error processing data for SKU {sku}: {e}", exc_info=True)
        return None

    if len(df_daily) < 14:
        logger.warning(f"Not enough aggregated data points ({len(df_daily)}) for SKU: {sku}. Need at least 14.")
        return None

    return df_daily

