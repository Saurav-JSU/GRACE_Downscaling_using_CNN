import os
import sys
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GRACEDataValidator:
    """Validator for GRACE TIFF data files"""
    
    def __init__(self, grace_dir: Path, output_dir: Path):
        """
        Initialize the validator
        
        Args:
            grace_dir: Directory containing GRACE TIFF files
            output_dir: Directory to save validation results
        """
        self.grace_dir = Path(grace_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store validation results
        self.available_months: List[str] = []
        self.data_stats: Dict = {}
        self.spatial_info: Dict = {}
        self.temporal_coverage = pd.DataFrame()
        
    def scan_grace_files(self) -> List[Path]:
        """Scan directory for GRACE TIFF files"""
        grace_files = list(self.grace_dir.glob("grace_lwe_*.tif"))
        grace_files.sort()
        
        if not grace_files:
            raise FileNotFoundError(f"No GRACE TIFF files found in {self.grace_dir}")
            
        logger.info(f"Found {len(grace_files)} GRACE TIFF files")
        return grace_files
        
    def validate_single_file(self, file_path: Path) -> Dict:
        """
        Validate a single GRACE TIFF file
        
        Args:
            file_path: Path to GRACE TIFF file
            
        Returns:
            Dict containing validation results
        """
        try:
            with rasterio.open(file_path) as src:
                # Read data
                data = src.read(1)  # Read first band
                
                # Get metadata
                metadata = src.meta
                
                # Calculate statistics
                stats = {
                    'min': float(np.nanmin(data)),
                    'max': float(np.nanmax(data)),
                    'mean': float(np.nanmean(data)),
                    'std': float(np.nanstd(data)),
                    'nan_count': int(np.isnan(data).sum()),
                    'resolution_x': float(src.res[0]),
                    'resolution_y': float(src.res[1]),
                    'crs': str(src.crs),
                    'shape': data.shape,
                    'bounds': src.bounds
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error validating {file_path}: {str(e)}")
            raise
            
    def parse_date_from_filename(self, filename: str) -> datetime:
        """Extract date from GRACE filename"""
        # Expected format: grace_lwe_YYYY_MM.tif
        try:
            parts = filename.split('_')
            year = int(parts[2])
            month = int(parts[3].replace('.tif', ''))
            return datetime(year, month, 1)
        except Exception as e:
            logger.error(f"Error parsing date from filename {filename}: {str(e)}")
            raise
            
    def analyze_temporal_coverage(self, files: List[Path]):
        """Analyze temporal coverage of GRACE data"""
        dates = []
        for file_path in files:
            date = self.parse_date_from_filename(file_path.name)
            dates.append(date)
            
        # Create date range for full period
        start_date = min(dates)
        end_date = max(dates)
        all_months = pd.date_range(start_date, end_date, freq='ME')
        
        # Create coverage DataFrame
        coverage = pd.DataFrame(index=all_months)
        coverage['has_data'] = coverage.index.isin(pd.to_datetime(dates))
        
        self.temporal_coverage = coverage
        
        # Store available dates for reference
        self.available_dates = sorted(dates)
        
        # Calculate coverage statistics
        total_months = len(all_months)
        available_months = len(dates)
        coverage_percent = (available_months / total_months) * 100
        
        logger.info(f"Temporal coverage: {coverage_percent:.1f}% ({available_months}/{total_months} months)")
        
        return coverage
        
    def validate_all_files(self):
        """Validate all GRACE TIFF files"""
        grace_files = self.scan_grace_files()
        logger.info(f"Found {len(grace_files)} GRACE TIFF files")
        
        # Initialize results storage
        all_results = []
        
        # Validate each file
        for file_path in grace_files:
            logger.info(f"Validating {file_path.name}")
            try:
                stats = self.validate_single_file(file_path)
                stats['filename'] = file_path.name
                stats['date'] = self.parse_date_from_filename(file_path.name)
                all_results.append(stats)
                logger.info(f"Successfully validated {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to validate {file_path.name}: {str(e)}")
                continue
                
        # Convert to DataFrame for analysis
        self.validation_df = pd.DataFrame(all_results)
        logger.info(f"Successfully validated {len(self.validation_df)} files")
        
        # Analyze temporal coverage
        self.analyze_temporal_coverage(grace_files)
        
    def generate_report(self):
        """Generate validation report"""
        report_path = self.output_dir / "grace_validation_report.txt"
        
        # Count available months directly from validation_df
        available_months = len(self.validation_df)
        total_months = len(self.temporal_coverage)
        coverage_percent = (available_months / total_months) * 100
        
        with open(report_path, 'w') as f:
            f.write("GRACE Data Validation Report\n")
            f.write("==========================\n\n")
            
            # Temporal coverage
            f.write("Temporal Coverage:\n")
            f.write("-----------------\n")
            f.write(f"Start Date: {self.temporal_coverage.index.min():%Y-%m}\n")
            f.write(f"End Date: {self.temporal_coverage.index.max():%Y-%m}\n")
            f.write(f"Total Months: {total_months}\n")
            f.write(f"Available Months: {available_months}\n")
            f.write(f"Coverage: {coverage_percent:.1f}%\n\n")
            
            # Spatial information
            f.write("Spatial Information:\n")
            f.write("-------------------\n")
            if len(self.validation_df) > 0:
                f.write(f"Resolution: {self.validation_df['resolution_x'].iloc[0]}° x ")
                f.write(f"{self.validation_df['resolution_y'].iloc[0]}°\n")
                f.write(f"CRS: {self.validation_df['crs'].iloc[0]}\n")
                f.write(f"Data Shape: {self.validation_df['shape'].iloc[0]}\n\n")
            
            # Data statistics
            f.write("Data Statistics:\n")
            f.write("---------------\n")
            f.write("Mean Values Range: ")
            f.write(f"{self.validation_df['mean'].min():.2f} to {self.validation_df['mean'].max():.2f}\n")
            f.write("Standard Deviation Range: ")
            f.write(f"{self.validation_df['std'].min():.2f} to {self.validation_df['std'].max():.2f}\n")
            f.write("Missing Data (NaN) Range: ")
            f.write(f"{self.validation_df['nan_count'].min()} to {self.validation_df['nan_count'].max()}\n")
            
        logger.info(f"Validation report saved to {report_path}")
        
    def plot_temporal_coverage(self):
        """Plot temporal coverage"""
        plt.figure(figsize=(12, 4))
        plt.plot(self.temporal_coverage.index, self.temporal_coverage['has_data'], 'b.')
        plt.grid(True)
        plt.title('GRACE Data Temporal Coverage')
        plt.xlabel('Date')
        plt.ylabel('Data Available')
        plt.savefig(self.output_dir / 'temporal_coverage.png')
        plt.close()
        
    def run_validation(self):
        """Run complete validation process"""
        logger.info("Starting GRACE data validation")
        
        try:
            # Validate all files
            self.validate_all_files()
            
            # Generate report
            self.generate_report()
            
            # Plot temporal coverage
            self.plot_temporal_coverage()
            
            logger.info("Validation completed successfully")
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise

def main():
    """Main function to run validation"""
    try:
        # Set up paths
        grace_dir = Path("Data/GRACE_tiff_data")  # Update this path
        output_dir = Path("validation_results")
        
        # Initialize and run validator
        validator = GRACEDataValidator(grace_dir, output_dir)
        validator.run_validation()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()