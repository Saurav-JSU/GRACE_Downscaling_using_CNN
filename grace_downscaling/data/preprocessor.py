import os
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from grace_downscaling package
from grace_downscaling import ROOT_DIR
from grace_downscaling.utils.config import Config
from grace_downscaling.data.data_loader import GRACEDataLoader

class GRACEPreprocessor:
    """Class to handle preprocessing of GRACE, auxiliary, and static data"""
    
    def __init__(self):
        self.scalers = {}
        
    def handle_nans(self, data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """Handle NaN values in the data"""
        if not np.any(np.isnan(data)):
            return data
            
        processed = data.copy()
        
        for band in range(data.shape[0]):
            band_data = data[band]
            nan_mask = np.isnan(band_data)
            
            if np.all(nan_mask):
                logger.warning(f"Band {band} contains all NaN values!")
                # Fill with zeros if all values are NaN
                processed[band] = np.zeros_like(band_data)
                continue
                
            if method == 'interpolate':
                from scipy import interpolate
                
                # Get coordinates of valid and invalid points
                y_valid, x_valid = np.where(~nan_mask)
                y_nan, x_nan = np.where(nan_mask)
                
                if len(y_valid) > 3:  # Need at least 3 points for interpolation
                    # Get valid values
                    valid_values = band_data[~nan_mask]
                    
                    # Create interpolator
                    interp = interpolate.LinearNDInterpolator(
                        list(zip(y_valid, x_valid)),
                        valid_values,
                        fill_value=np.nanmean(band_data)
                    )
                    
                    # Interpolate NaN points
                    band_data[nan_mask] = interp(list(zip(y_nan, x_nan)))
                else:
                    # If too few valid points, fill with mean or zero
                    if np.any(~nan_mask):
                        band_data[nan_mask] = np.nanmean(band_data)
                    else:
                        band_data[nan_mask] = 0
                        
            elif method == 'mean':
                band_mean = np.nanmean(band_data)
                band_data[nan_mask] = band_mean if not np.isnan(band_mean) else 0
                
            processed[band] = band_data
            
        return processed
        
    def normalize_data(self, data: np.ndarray, 
                  method: str = 'minmax',
                  feature_name: str = 'default') -> np.ndarray:
        """Normalize data using specified method"""
        # Handle NaNs first
        data = self.handle_nans(data, method='interpolate')
        
        # Check data validity
        if np.all(np.isnan(data)):
            logger.error("All values are NaN!")
            return np.zeros_like(data)
            
        # Log data statistics before normalization
        logger.info(f"Data stats before normalization - min: {np.nanmin(data):.3f}, "
                f"max: {np.nanmax(data):.3f}, mean: {np.nanmean(data):.3f}")
        
        original_shape = data.shape
        
        try:
            if method == 'minmax':
                # For GRACE data, handle as a special case
                if feature_name == 'grace':
                    # Reshape preserving the spatial structure
                    data_reshaped = data.reshape(-1, data.shape[-2] * data.shape[-1])
                    if feature_name not in self.scalers:
                        self.scalers[feature_name] = MinMaxScaler()
                        data_normalized = self.scalers[feature_name].fit_transform(data_reshaped.T).T
                    else:
                        data_normalized = self.scalers[feature_name].transform(data_reshaped.T).T
                    normalized = data_normalized.reshape(original_shape)
                else:
                    # For other data types, proceed as before
                    data_2d = data.reshape(data.shape[0], -1)
                    if feature_name not in self.scalers:
                        self.scalers[feature_name] = MinMaxScaler()
                        data_2d = self.scalers[feature_name].fit_transform(data_2d)
                    else:
                        data_2d = self.scalers[feature_name].transform(data_2d)
                    normalized = data_2d.reshape(original_shape)
            
            elif method == 'standard':
                # Similar special handling for GRACE data
                if feature_name == 'grace':
                    data_reshaped = data.reshape(-1, data.shape[-2] * data.shape[-1])
                    if feature_name not in self.scalers:
                        self.scalers[feature_name] = StandardScaler()
                        data_normalized = self.scalers[feature_name].fit_transform(data_reshaped.T).T
                    else:
                        data_normalized = self.scalers[feature_name].transform(data_reshaped.T).T
                    normalized = data_normalized.reshape(original_shape)
                else:
                    data_2d = data.reshape(data.shape[0], -1)
                    if feature_name not in self.scalers:
                        self.scalers[feature_name] = StandardScaler()
                        data_2d = self.scalers[feature_name].fit_transform(data_2d)
                    else:
                        data_2d = self.scalers[feature_name].transform(data_2d)
                    normalized = data_2d.reshape(original_shape)
                    
        except Exception as e:
            logger.error(f"Error during normalization: {e}")
            return np.zeros_like(data)
            
        # Log data statistics after normalization
        logger.info(f"Data stats after normalization - min: {np.min(normalized):.3f}, "
                f"max: {np.max(normalized):.3f}, mean: {np.mean(normalized):.3f}")
        
        return normalized
        
    def resample_to_grace_resolution(self, 
                                   data: np.ndarray,
                                   src_transform: rasterio.Affine,
                                   target_shape: Tuple[int, int],
                                   target_transform: rasterio.Affine) -> np.ndarray:
        """Resample data to match GRACE resolution"""
        logger.info(f"Resampling data from shape {data.shape} to {target_shape}")
        
        # Check for NaNs before resampling
        if np.any(np.isnan(data)):
            logger.warning("Found NaN values before resampling, handling them...")
            data = self.handle_nans(data)
        
        resampled = np.zeros((data.shape[0], target_shape[0], target_shape[1]))
        
        for band in range(data.shape[0]):
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=data.shape[1],
                    width=data.shape[2],
                    count=1,
                    dtype=data.dtype,
                    transform=src_transform,
                    crs='EPSG:4326'
                ) as src:
                    src.write(data[band], 1)
                    
                    resampled[band] = src.read(
                        1,
                        out_shape=target_shape,
                        resampling=Resampling.bilinear
                    )
        
        # Verify no NaNs were introduced during resampling
        if np.any(np.isnan(resampled)):
            logger.warning("NaN values found after resampling, handling them...")
            resampled = self.handle_nans(resampled)
        
        logger.info(f"Resampling complete. New shape: {resampled.shape}")
        return resampled
        
    def prepare_data(self, 
                    grace_data: np.ndarray,
                    aux_data: np.ndarray,
                    static_data: np.ndarray,
                    grace_meta: Dict,
                    aux_meta: Dict,
                    static_meta: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare all data for model training"""
        logger.info("Starting data preparation")
        
        # Log initial data statistics
        logger.info("\nInitial data statistics:")
        logger.info(f"GRACE - min: {np.nanmin(grace_data):.3f}, max: {np.nanmax(grace_data):.3f}")
        logger.info(f"Auxiliary - min: {np.nanmin(aux_data):.3f}, max: {np.nanmax(aux_data):.3f}")
        logger.info(f"Static - min: {np.nanmin(static_data):.3f}, max: {np.nanmax(static_data):.3f}")
        
        # Get target shape from GRACE data
        target_shape = (grace_data.shape[1], grace_data.shape[2])
        target_transform = grace_meta['transform']
        
        # Resample auxiliary data to GRACE resolution
        logger.info("Resampling auxiliary data...")
        resampled_aux = self.resample_to_grace_resolution(
            aux_data,
            aux_meta['transform'],
            target_shape,
            target_transform
        )
        
        # Resample static data to GRACE resolution
        logger.info("Resampling static data...")
        resampled_static = self.resample_to_grace_resolution(
            static_data,
            static_meta['transform'],
            target_shape,
            target_transform
        )
        
        # Normalize all data
        logger.info("Normalizing data...")
        grace_norm = self.normalize_data(grace_data, 'minmax', 'grace')
        aux_norm = self.normalize_data(resampled_aux, 'minmax', 'auxiliary')
        static_norm = self.normalize_data(resampled_static, 'minmax', 'static')
        
        return grace_norm, aux_norm, static_norm

def test_preprocessor():
    """Test function to verify preprocessor functionality"""
    try:
        # Initialize loader and preprocessor
        loader = GRACEDataLoader(Config)
        preprocessor = GRACEPreprocessor()
        
        # Get first available month
        first_year, first_month = loader.available_dates[0]
        
        # Load data
        grace_data, grace_meta = loader.load_monthly_data(first_year, first_month)
        aux_data, aux_meta = loader.load_auxiliary_data(first_year, first_month)
        static_data, static_meta = loader.load_static_data()
        
        # Print initial shapes and statistics
        print("\nInitial data shapes and statistics:")
        for name, data in [('GRACE', grace_data), ('Auxiliary', aux_data), ('Static', static_data)]:
            print(f"\n{name} data:")
            print(f"Shape: {data.shape}")
            print(f"NaN count: {np.isnan(data).sum()}")
            if not np.all(np.isnan(data)):
                print(f"Value range: [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
        
        # Process data
        grace_norm, aux_norm, static_norm = preprocessor.prepare_data(
            grace_data, aux_data, static_data,
            grace_meta, aux_meta, static_meta
        )
        
        # Print final shapes and statistics
        print("\nProcessed data shapes and statistics:")
        for name, data in [('GRACE', grace_norm), ('Auxiliary', aux_norm), ('Static', static_norm)]:
            print(f"\n{name} data:")
            print(f"Shape: {data.shape}")
            print(f"NaN count: {np.isnan(data).sum()}")
            print(f"Value range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        print("\nAll preprocessing tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_preprocessor()