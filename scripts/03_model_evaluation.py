import os
import sys
import logging
import numpy as np
import tensorflow as tf
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

# Add the project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from grace_downscaling.utils.config import Config
from grace_downscaling.data.data_loader import GRACEDataLoader
from grace_downscaling.data.preprocessor import GRACEPreprocessor
from grace_downscaling.models.cnn_model import GRACEDownscalingModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_dir: Path):
        self.config = Config
        self.loader = GRACEDataLoader(self.config)
        self.preprocessor = GRACEPreprocessor()
        self.model_dir = model_dir
        
        # Create output directory
        self.output_dir = model_dir / "evaluation"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Load trained stage 1 and stage 2 models"""
        # Find the best model files
        stage1_files = list(self.model_dir.glob("stage1_model_*.keras"))
        stage2_files = list(self.model_dir.glob("stage2_model_*.keras"))
        
        # Sort by validation loss (encoded in filename)
        stage1_files.sort(key=lambda x: float(str(x).split('-')[-1].replace('.keras', '')))
        stage2_files.sort(key=lambda x: float(str(x).split('-')[-1].replace('.keras', '')))
        
        # Load best models
        stage1_model = tf.keras.models.load_model(str(stage1_files[0]))
        stage2_model = tf.keras.models.load_model(str(stage2_files[0]))
        
        logger.info(f"Loaded stage 1 model from: {stage1_files[0]}")
        logger.info(f"Loaded stage 2 model from: {stage2_files[0]}")
        
        return stage1_model, stage2_model
    
    def downscale_data(self, year: int, month: int) -> Dict:
        """Downscale data for a specific month"""
        # Load and preprocess data
        monthly_data, monthly_meta = self.loader.load_monthly_data(year, month)
        static_data, static_meta = self.loader.load_static_data()
        
        # Preprocess
        monthly_patches, static_patches = self.preprocessor.prepare_training_data(
            monthly_data, static_data, monthly_meta, static_meta
        )
        
        # Reshape for model input (N, H, W, C)
        monthly_input = np.transpose(monthly_patches, (0, 2, 3, 1))
        static_input = np.transpose(static_patches, (0, 2, 3, 1))
        
        # Load models
        stage1_model, stage2_model = self.load_models()
        
        # Stage 1 downscaling (55km → 25km)
        stage1_output = stage1_model.predict([monthly_input, static_input])
        
        # Stage 2 downscaling (25km → 10km)
        stage2_output = stage2_model.predict([monthly_input, static_input])
        
        # Store metadata for saving
        transform = monthly_meta['transform']
        new_transform = rasterio.Affine(
            transform.a / 4, transform.b, transform.c,
            transform.d, transform.e / 4, transform.f
        )
        
        return {
            'original': monthly_patches[0, 0],  # First patch, first channel
            'stage1': stage1_output[0, ..., 0],  # First patch
            'stage2': stage2_output[0, ..., 0],  # First patch
            'original_transform': transform,
            'new_transform': new_transform,
            'crs': monthly_meta['crs']
        }
    
    def save_results(self, results: Dict, year: int, month: int):
        """Save results as GeoTIFF files"""
        timestamp = f"{year}_{month:02d}"
        
        # Save original resolution
        with rasterio.open(
            self.output_dir / f"original_{timestamp}.tif",
            'w',
            driver='GTiff',
            height=results['original'].shape[0],
            width=results['original'].shape[1],
            count=1,
            dtype=results['original'].dtype,
            crs=results['crs'],
            transform=results['original_transform']
        ) as dst:
            dst.write(results['original'], 1)
        
        # Save stage 1 result (25km)
        with rasterio.open(
            self.output_dir / f"downscaled_25km_{timestamp}.tif",
            'w',
            driver='GTiff',
            height=results['stage1'].shape[0],
            width=results['stage1'].shape[1],
            count=1,
            dtype=results['stage1'].dtype,
            crs=results['crs'],
            transform=results['new_transform']
        ) as dst:
            dst.write(results['stage1'], 1)
        
        # Save stage 2 result (10km)
        with rasterio.open(
            self.output_dir / f"downscaled_10km_{timestamp}.tif",
            'w',
            driver='GTiff',
            height=results['stage2'].shape[0],
            width=results['stage2'].shape[1],
            count=1,
            dtype=results['stage2'].dtype,
            crs=results['crs'],
            transform=results['new_transform']
        ) as dst:
            dst.write(results['stage2'], 1)
            
    def visualize_results(self, results: Dict, year: int, month: int):
        """Create visualization of the results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original resolution
        im0 = axes[0].imshow(results['original'])
        axes[0].set_title(f'Original ({55}km)')
        plt.colorbar(im0, ax=axes[0])
        
        # Stage 1 (25km)
        im1 = axes[1].imshow(results['stage1'])
        axes[1].set_title(f'Stage 1 (25km)')
        plt.colorbar(im1, ax=axes[1])
        
        # Stage 2 (10km)
        im2 = axes[2].imshow(results['stage2'])
        axes[2].set_title(f'Stage 2 (10km)')
        plt.colorbar(im2, ax=axes[2])
        
        plt.suptitle(f'GRACE Data Downscaling Results - {year}-{month:02d}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f'comparison_{year}_{month:02d}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main evaluation function"""
    try:
        # Find latest model run directory
        model_dirs = list(Path(Config.BASE_DIR / "models").glob("run_*"))
        latest_run = max(model_dirs, key=lambda x: str(x))
        
        logger.info(f"Using model directory: {latest_run}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(latest_run)
        
        # Process test months
        test_months = [
            (2013, 12),  # Last month
            (2008, 6),   # Middle month
            (2002, 1)    # First month
        ]
        
        for year, month in test_months:
            logger.info(f"Processing {year}-{month:02d}")
            
            # Downscale data
            results = evaluator.downscale_data(year, month)
            
            # Save results
            evaluator.save_results(results, year, month)
            
            # Create visualization
            evaluator.visualize_results(results, year, month)
            
            logger.info(f"Completed processing {year}-{month:02d}")
            
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()