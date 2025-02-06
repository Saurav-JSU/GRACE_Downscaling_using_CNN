import os
import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from grace_downscaling package
from grace_downscaling import ROOT_DIR
from grace_downscaling.utils.config import Config

# Add the project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from grace_downscaling.utils.config import Config
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current directory: {current_dir}")
    raise

class GRACEDataLoader:
    """
    Class to handle loading and basic processing of GRACE and auxiliary data
    """
    def __init__(self, config: Config):
        """
        Initialize the data loader with caching
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        self._static_data_cache = None
        self._aux_data_cache = {}  # Cache for auxiliary data
        self.available_dates = []
        
        # Initialize caches and validate paths
        self._validate_config()
        self._scan_available_months()
        
        # Pre-load static data
        logger.info("Pre-loading static data...")
        self.load_static_data()
        logger.info("Static data loaded and cached")
        
    def _validate_config(self):
        """Validate configuration and paths"""
        try:
            required_dirs = [
                self.config.DATA_DIR,
                self.config.DYNAMIC_DATA_DIR,
                self.config.STATIC_DATA_DIR,
                self.config.GRACE_DATA_DIR,
                self.config.STATION_DATA_DIR
            ]
            
            required_files = [
                self.config.STATIC_DATA_FILE,
                self.config.GROUNDWATER_DATA_FILE,
                self.config.STATION_COORDS_FILE
            ]
            
            # Check directories
            for directory in required_dirs:
                if not directory.exists():
                    raise FileNotFoundError(f"Required directory not found: {directory}")
            
            # Check files
            for file_path in required_files:
                if not file_path.exists():
                    raise FileNotFoundError(f"Required file not found: {file_path}")
                    
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
            
    def _scan_available_months(self):
        """Scan GRACE directory for available months"""
        grace_files = list(self.config.GRACE_DATA_DIR.glob("grace_lwe_*.tif"))
        
        # Extract dates from filenames
        self.available_dates = []
        for file_path in grace_files:
            try:
                # Parse date from filename (format: grace_lwe_YYYY_MM.tif)
                parts = file_path.stem.split('_')
                year = int(parts[2])
                month = int(parts[3])
                self.available_dates.append((year, month))
            except Exception as e:
                logger.warning(f"Could not parse date from {file_path.name}: {e}")
                
        self.available_dates.sort()
        logger.info(f"Found {len(self.available_dates)} available GRACE data files")
            
    def is_data_available(self, year: int, month: int) -> bool:
        """Check if GRACE data is available for given year and month"""
        return (year, month) in self.available_dates
            
    def load_monthly_data(self, year: int, month: int) -> Tuple[np.ndarray, Dict]:
        """
        Load monthly GRACE data
        
        Args:
            year: Year to load
            month: Month to load
            
        Returns:
            Tuple[np.ndarray, Dict]: Monthly data array and metadata
        """
        if not self.is_data_available(year, month):
            raise FileNotFoundError(f"No GRACE data available for {year}-{month:02d}")
            
        try:
            file_path = self.config.GRACE_DATA_DIR / f"grace_lwe_{year}_{month:02d}.tif"
            
            with rasterio.open(file_path) as src:
                # Read data and reshape to (channels, height, width)
                grace_data = src.read()  # Already in (bands, height, width) format
                metadata = src.meta.copy()
                
                logger.info(f"Loaded GRACE data for {year}-{month:02d} with shape {grace_data.shape}")
                
                return grace_data, metadata
                
        except Exception as e:
            logger.error(f"Error loading GRACE data for {year}-{month:02d}: {e}")
            raise

    def _clear_cache(self):
        """Clear the data caches to free memory"""
        self._static_data_cache = None
        self._aux_data_cache.clear()
        logger.info("Cleared data caches")
            
    def load_static_data(self) -> Tuple[np.ndarray, Dict]:
        """
        Load static data with caching
        
        Returns:
            Tuple[np.ndarray, Dict]: Static data array and metadata
        """
        try:
            # Return cached data if available
            if self._static_data_cache is not None:
                return self._static_data_cache
            
            # Load and cache if not available
            with rasterio.open(self.config.STATIC_DATA_FILE) as src:
                static_data = src.read()
                metadata = src.meta.copy()
                self._static_data_cache = (static_data, metadata)
                logger.info(f"Loaded and cached static data with shape: {static_data.shape}")
                return self._static_data_cache
                
        except Exception as e:
            logger.error(f"Error loading static data: {e}")
            raise
            
    def load_auxiliary_data(self, year: int, month: int) -> Tuple[np.ndarray, Dict]:
        """
        Load auxiliary monthly data with caching
        
        Args:
            year: Year to load
            month: Month to load
            
        Returns:
            Tuple[np.ndarray, Dict]: Auxiliary data array and metadata
        """
        cache_key = f"{year}_{month:02d}"
        
        try:
            # Return cached data if available
            if cache_key in self._aux_data_cache:
                logger.info(f"Using cached auxiliary data for {year}-{month:02d}")
                return self._aux_data_cache[cache_key]
            
            # Load and cache if not available
            file_path = self.config.DYNAMIC_DATA_DIR / f"monthly_data_{year}_{month:02d}.tif"
            
            with rasterio.open(file_path) as src:
                aux_data = src.read()
                metadata = src.meta.copy()
                
                # Cache the data
                self._aux_data_cache[cache_key] = (aux_data, metadata)
                
                logger.info(f"Loaded and cached auxiliary data for {year}-{month:02d} with shape: {aux_data.shape}")
                return aux_data, metadata
                
        except Exception as e:
            logger.error(f"Error loading auxiliary data for {year}-{month:02d}: {e}")
            raise
            
    def load_groundwater_data(self) -> pd.DataFrame:
        """
        Load and process groundwater station data
        
        Returns:
            pd.DataFrame: Processed groundwater data
        """
        try:
            df = pd.read_csv(self.config.GROUNDWATER_DATA_FILE)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded groundwater data with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading groundwater data: {e}")
            raise
            
    def load_station_coordinates(self) -> pd.DataFrame:
        """
        Load station coordinates
        
        Returns:
            pd.DataFrame: Station coordinates data
        """
        try:
            df = pd.read_csv(self.config.STATION_COORDS_FILE)
            logger.info(f"Loaded station coordinates with {len(df)} stations")
            return df
            
        except Exception as e:
            logger.error(f"Error loading station coordinates: {e}")
            raise

def test_data_loader():
    """Test function to verify data loader functionality"""
    try:
        # Initialize loader
        loader = GRACEDataLoader(Config)
        
        # Print available dates
        print("\nAvailable GRACE data months:")
        for year, month in loader.available_dates[:5]:
            print(f"{year}-{month:02d}")
        print("...")
        
        # Test loading first available month
        if loader.available_dates:
            first_year, first_month = loader.available_dates[0]
            
            # Test GRACE data loading
            grace_data, grace_meta = loader.load_monthly_data(first_year, first_month)
            print(f"\nGRACE data shape: {grace_data.shape}")
            print(f"GRACE metadata: {grace_meta}")
            
            # Test auxiliary data loading
            aux_data, aux_meta = loader.load_auxiliary_data(first_year, first_month)
            print(f"\nAuxiliary data shape: {aux_data.shape}")
            print(f"Number of auxiliary variables: {aux_data.shape[0]}")
        
        # Test static data loading
        static_data, static_meta = loader.load_static_data()
        print(f"\nStatic data shape: {static_data.shape}")
        print(f"Number of static variables: {static_data.shape[0]}")
        
        # Test groundwater data loading
        gw_data = loader.load_groundwater_data()
        print(f"\nGroundwater data shape: {gw_data.shape}")
        print(f"Groundwater data columns: {gw_data.columns.tolist()}")
        
        # Test station coordinates loading
        station_coords = loader.load_station_coordinates()
        print(f"\nStation coordinates shape: {station_coords.shape}")
        
        print("\nAll data loading tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_data_loader()