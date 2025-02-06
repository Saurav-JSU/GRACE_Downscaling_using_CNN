import os
import sys
import logging
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from grace_downscaling.utils.config import Config
from grace_downscaling.data.data_generator import DataGeneratorBuilder
from grace_downscaling.models.cnn_model import GRACEDownscalingModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_training_directories():
    """Create necessary directories for model artifacts"""
    base_dir = Path(Config.BASE_DIR)
    model_dir = base_dir / "models"
    log_dir = base_dir / "logs"
    
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    return run_dir

def create_callbacks(run_dir: Path, stage: int):
    """Create training callbacks"""
    checkpoint_path = run_dir / f"stage{stage}_model_{{epoch:02d}}-{{val_loss:.4f}}.keras"
    log_path = run_dir / 'logs'
    csv_path = run_dir / f'stage{stage}_training_log.csv'
    
    log_path.mkdir(exist_ok=True)
    
    callbacks = [
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        ),
        # Learning rate reduction on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # CSV logging
        tf.keras.callbacks.CSVLogger(
            str(csv_path),
            append=True
        )
    ]
    return callbacks

def train_stage(stage: int, data_generator_builder: DataGeneratorBuilder, run_dir: Path):
    """Train a single stage of the model"""
    logger.info(f"Starting training for Stage {stage}")
    
    # Get data generators
    train_gen, val_gen = data_generator_builder.build_generators(
        years=list(range(Config.START_YEAR, Config.END_YEAR + 1)),
        months=list(range(1, 13)),
        batch_size=Config.BATCH_SIZE,
        val_split=Config.VALIDATION_SPLIT,
        stage=stage
    )
    
    # Get input shapes from the generator
    monthly_shape = train_gen.monthly_patches.shape[1:]
    static_shape = train_gen.static_patches.shape[1:]
    
    # Initialize model
    model = GRACEDownscalingModel(
        input_shape_monthly=monthly_shape,
        input_shape_static=static_shape,
        learning_rate=Config.LEARNING_RATE
    )
    
    # Get the appropriate model based on stage
    if stage == 1:
        model = model.build_stage1_model()
        logger.info("Stage 1 model built successfully")
    else:
        model = model.build_stage2_model()
        logger.info("Stage 2 model built successfully")
    
    # Create callbacks
    callbacks = create_callbacks(run_dir, stage)
    
    # Train the model
    logger.info(f"Starting training for {Config.EPOCHS} epochs")
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.EPOCHS,
            callbacks=callbacks
        )
        
        # Save final model
        final_model_path = run_dir / f"stage{stage}_final_model.keras"
        model.save(str(final_model_path))
        logger.info(f"Stage {stage} training completed. Model saved at {final_model_path}")
        
        return history, model
        
    except Exception as e:
        logger.error(f"Error during training stage {stage}: {str(e)}")
        raise

def main():
    """Main training function"""
    try:
        # Create directories
        run_dir = setup_training_directories()
        logger.info(f"Created run directory at {run_dir}")
        
        # Initialize data generator builder
        data_generator_builder = DataGeneratorBuilder(Config)
        
        # Train Stage 1 (55km to 25km)
        logger.info("Starting Stage 1 training")
        history_stage1, model_stage1 = train_stage(1, data_generator_builder, run_dir)
        
        # Train Stage 2 (25km to 10km)
        logger.info("Starting Stage 2 training")
        history_stage2, model_stage2 = train_stage(2, data_generator_builder, run_dir)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()