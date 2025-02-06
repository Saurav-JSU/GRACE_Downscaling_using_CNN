import os
import sys
import tensorflow as tf
import numpy as np
import logging
import psutil
import gc
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, Tuple
import json

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CPU Configuration
TOTAL_CORES = 120
SYSTEM_CORES = 4
TRAINING_CORES = TOTAL_CORES - SYSTEM_CORES

# Configure TensorFlow threading
tf.config.threading.set_inter_op_parallelism_threads(TRAINING_CORES // 4)
tf.config.threading.set_intra_op_parallelism_threads(TRAINING_CORES // 2)

# Set memory growth - important for CPU training
physical_devices = tf.config.list_physical_devices('CPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Import custom modules
from grace_downscaling.utils.config import Config
from grace_downscaling.data.optimized_generator import OptimizedGRACEGenerator
from grace_downscaling.models.cnn_model import GRACEDownscalingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Training configurations
BATCH_SIZE = 64  # Optimized for CPU
EPOCHS = 100
BASE_LR = 1e-4  # Lower learning rate for stability
VAL_FREQ = 1  # Validate every N epochs
MEMORY_LIMIT_GB = 64  # Memory limit per generator

class TrainingManager:
    """Manages the training process for GRACE downscaling"""
    
    def __init__(self, run_dir: Path):
        """Initialize training manager"""
        self.run_dir = run_dir
        self.setup_directories()
        self.memory_monitor = psutil.Process()
        self.training_history = {'stage1': {}, 'stage2': {}}
    
    def setup_directories(self):
        """Setup necessary directories"""
        self.dirs = {
            'checkpoints': self.run_dir / 'checkpoints',
            'logs': self.run_dir / 'logs',
            'cache': self.run_dir / 'cache',
            'results': self.run_dir / 'results'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing cache
        if self.dirs['cache'].exists():
            shutil.rmtree(self.dirs['cache'])
            self.dirs['cache'].mkdir()
    
    def get_callbacks(self, stage: int) -> list:
        """Get callbacks for training"""
        checkpoint_path = self.dirs['checkpoints'] / f'stage{stage}'
        log_path = self.dirs['logs'] / f'stage{stage}'
        
        callbacks = [
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path / 'model_epoch_{epoch:02d}.keras'),
                save_weights_only=True,
                save_freq='epoch'
            ),
            # Best model saving
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path / 'best_model.keras'),
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            # Learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                cooldown=3
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=str(log_path),
                update_freq='epoch',
                profile_batch=0
            ),
            # CSV logging
            tf.keras.callbacks.CSVLogger(
                str(log_path / 'training_log.csv'),
                append=True
            ),
            # Memory monitoring
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logger.info(
                    f"Memory usage: {self.memory_monitor.memory_info().rss / 1024**3:.2f}GB"
                )
            )
        ]
        
        return callbacks
    
    def create_generators(self, stage: int, train_years: list, val_years: list) -> Tuple:
        """Create data generators for training and validation"""
        train_gen = OptimizedGRACEGenerator(
            years=train_years,
            months=range(1, 13),
            batch_size=BATCH_SIZE,
            stage=stage,
            shuffle=True,
            is_training=True,
            cache_dir=self.dirs['cache'] / f'stage{stage}' / 'train',
            max_memory_gb=MEMORY_LIMIT_GB
        )
        
        val_gen = OptimizedGRACEGenerator(
            years=val_years,
            months=range(1, 13),
            batch_size=BATCH_SIZE,
            stage=stage,
            shuffle=False,
            is_training=False,
            cache_dir=self.dirs['cache'] / f'stage{stage}' / 'val',
            max_memory_gb=MEMORY_LIMIT_GB
        )
        
        return train_gen, val_gen
    
    def train_stage(self, stage: int, train_gen, val_gen) -> Dict:
        """Train a single stage"""
        logger.info(f"\nStarting Stage {stage} Training")
        
        # Create model
        model = GRACEDownscalingModel(learning_rate=BASE_LR)
        if stage == 1:
            model = model.build_stage1_model()
        else:
            model = model.build_stage2_model()
        
        # Train model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=self.get_callbacks(stage),
            workers=TRAINING_CORES // 6,
            use_multiprocessing=True,
            max_queue_size=12,
            validation_freq=VAL_FREQ
        )
        
        # Save training history
        self.training_history[f'stage{stage}'] = history.history
        
        # Save model architecture
        model_json = model.to_json()
        with open(self.dirs['results'] / f'model_stage{stage}_architecture.json', 'w') as f:
            json.dump(model_json, f)
        
        return history.history
    
    def train_model(self) -> Dict:
        """Execute complete training pipeline"""
        try:
            # Data split
            train_years = list(range(2002, 2012))
            val_years = [2012, 2013]
            
            # Stage 1 Training
            logger.info("\nInitializing Stage 1")
            train_gen_s1, val_gen_s1 = self.create_generators(1, train_years, val_years)
            history_s1 = self.train_stage(1, train_gen_s1, val_gen_s1)
            
            # Clear memory before Stage 2
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Stage 2 Training
            logger.info("\nInitializing Stage 2")
            train_gen_s2, val_gen_s2 = self.create_generators(2, train_years, val_years)
            history_s2 = self.train_stage(2, train_gen_s2, val_gen_s2)
            
            # Save complete training history
            with open(self.dirs['results'] / 'training_history.json', 'w') as f:
                json.dump(self.training_history, f)
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Print system information
        logger.info("System Configuration:")
        logger.info(f"Total CPU Cores: {TOTAL_CORES}")
        logger.info(f"Training Cores: {TRAINING_CORES}")
        logger.info(f"Total Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
        logger.info(f"Batch Size: {BATCH_SIZE}")
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Config.MODEL_DIR / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'base_lr': BASE_LR,
            'memory_limit_gb': MEMORY_LIMIT_GB,
            'training_cores': TRAINING_CORES,
            'timestamp': timestamp
        }
        with open(run_dir / 'training_config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        # Initialize and run training
        trainer = TrainingManager(run_dir)
        history = trainer.train_model()
        
        logger.info(f"\nTraining completed successfully!")
        logger.info(f"Results saved in: {run_dir}")
        
        return history, run_dir
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()