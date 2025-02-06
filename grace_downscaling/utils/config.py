import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path("C:/Users/J01013381/OneDrive - Jackson State University/Research Projects/2025/ORISE/GW_Downscale/Approach3")
    DATA_DIR = BASE_DIR / "Data"
    CODE_DIR = BASE_DIR / "Code"
    
    # Model and log directories
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Data paths
    DYNAMIC_DATA_DIR = DATA_DIR / "Dynamic Monthly Data"
    STATIC_DATA_DIR = DATA_DIR / "Static Data"
    GRACE_DATA_DIR = DATA_DIR / "GRACE_tiff_data"
    STATION_DATA_DIR = DATA_DIR / "Station Data"
    
    # File paths
    STATIC_DATA_FILE = STATIC_DATA_DIR / "static_data.tif"
    GROUNDWATER_DATA_FILE = STATION_DATA_DIR / "groundwater_data.csv"
    STATION_COORDS_FILE = STATION_DATA_DIR / "station_coordinates.csv"
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Data parameters
    START_YEAR = 2002
    END_YEAR = 2013
    INPUT_RESOLUTION = 55  # km
    STAGE1_TARGET_RESOLUTION = 25  # km
    STAGE2_TARGET_RESOLUTION = 10  # km
    
    @classmethod
    def validate_paths(cls):
        """Validate that all necessary paths exist"""
        # Create directories if they don't exist
        directories = [
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.CACHE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Check required data directories
        required_dirs = [
            cls.DATA_DIR,
            cls.DYNAMIC_DATA_DIR,
            cls.STATIC_DATA_DIR,
            cls.GRACE_DATA_DIR,
            cls.STATION_DATA_DIR
        ]
        
        # Check required files
        required_files = [
            cls.STATIC_DATA_FILE,
            cls.GROUNDWATER_DATA_FILE,
            cls.STATION_COORDS_FILE
        ]
        
        # Validate directories
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Required directory not found: {directory}")
        
        # Validate files
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
                
    @classmethod
    def create_run_directories(cls) -> Path:
        """Create run-specific directories"""
        cls.validate_paths()
        return cls.MODEL_DIR

if __name__ == "__main__":
    # Test the configuration
    try:
        Config.validate_paths()
        print("All required paths exist!")
        
        # Print directory structure
        print("\nDirectory paths:")
        print(f"Base directory: {Config.BASE_DIR}")
        print(f"Model directory: {Config.MODEL_DIR}")
        print(f"Log directory: {Config.LOG_DIR}")
        print(f"Cache directory: {Config.CACHE_DIR}")
        
        print("\nData paths:")
        print(f"GRACE data directory: {Config.GRACE_DATA_DIR}")
        print(f"Static data file: {Config.STATIC_DATA_FILE}")
        
        print("\nModel parameters:")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        print(f"Epochs: {Config.EPOCHS}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")