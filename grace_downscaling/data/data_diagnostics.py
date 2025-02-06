import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from grace_downscaling.utils.config import Config
from grace_downscaling.data.data_loader import GRACEDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_data_quality(data: np.ndarray, data_name: str):
    """Analyze data quality including NaN values and basic statistics"""
    logger.info(f"\nAnalyzing {data_name}:")
    logger.info(f"Shape: {data.shape}")
    
    # Check for NaN values
    nan_count = np.isnan(data).sum()
    total_elements = data.size
    nan_percentage = (nan_count / total_elements) * 100
    
    logger.info(f"Total NaN values: {nan_count}")
    logger.info(f"Percentage of NaN values: {nan_percentage:.2f}%")
    
    # If there are NaN values, analyze their distribution across bands
    if nan_count > 0:
        logger.info("\nNaN distribution across bands:")
        for band in range(data.shape[0]):
            band_nan_count = np.isnan(data[band]).sum()
            band_nan_percentage = (band_nan_count / data[band].size) * 100
            logger.info(f"Band {band}: {band_nan_count} NaN values ({band_nan_percentage:.2f}%)")
    
    # Basic statistics excluding NaN values
    logger.info("\nBasic statistics (excluding NaN values):")
    logger.info(f"Min value: {np.nanmin(data)}")
    logger.info(f"Max value: {np.nanmax(data)}")
    logger.info(f"Mean value: {np.nanmean(data)}")
    logger.info(f"Std deviation: {np.nanstd(data)}")

def main():
    """Main function to run diagnostics"""
    try:
        # Initialize data loader
        loader = GRACEDataLoader(Config)
        
        # Load and analyze static data
        static_data, static_meta = loader.load_static_data()
        analyze_data_quality(static_data, "Static Data")
        
        # Load and analyze monthly data
        monthly_data, monthly_meta = loader.load_monthly_data(2002, 1)
        analyze_data_quality(monthly_data, "Monthly Data (2002-01)")
        
        # Additional metadata analysis
        logger.info("\nStatic Data Metadata:")
        for key, value in static_meta.items():
            logger.info(f"{key}: {value}")
            
        logger.info("\nMonthly Data Metadata:")
        for key, value in monthly_meta.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        raise

if __name__ == "__main__":
    main()