import os
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
import psutil
from functools import partial

from grace_downscaling.utils.config import Config
from grace_downscaling.data.data_loader import GRACEDataLoader
from grace_downscaling.data.preprocessor import GRACEPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedGRACEGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator for GRACE downscaling"""
    
    def __init__(self,
                 years: List[int],
                 months: List[int],
                 batch_size: int = 32,
                 stage: int = 1,
                 shuffle: bool = True,
                 is_training: bool = True,
                 cache_dir: Optional[str] = None,
                 max_memory_gb: float = 32.0):
        """
        Initialize the generator with memory constraints
        
        Args:
            years: List of years to process
            months: List of months to process
            batch_size: Batch size for training
            stage: Training stage (1 or 2)
            shuffle: Whether to shuffle data
            is_training: Whether this is for training
            cache_dir: Directory for caching processed data
            max_memory_gb: Maximum memory usage in GB
        """
        self.years = years
        self.months = months
        self.batch_size = batch_size
        self.stage = stage
        self.shuffle = shuffle
        self.is_training = is_training
        self.max_memory_gb = max_memory_gb
        
        # Initialize components
        self.loader = GRACEDataLoader(Config)
        self.preprocessor = GRACEPreprocessor()
        
        # Set up cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Config.CACHE_DIR / f"stage{stage}" / ("train" if is_training else "val")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data management
        self.available_months = self._get_available_months()
        self.num_samples = len(self.available_months)
        self.indices = np.arange(self.num_samples)
        self._data_cache = {}
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Calculate memory-efficient chunk size
        self.chunk_size = self._calculate_chunk_size()
        logger.info(f"Initialized generator with {self.num_samples} samples, chunk size: {self.chunk_size}")
        
    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on memory constraints"""
        # Estimate memory per sample (in GB)
        sample_size_gb = 0.1  # Approximate size for one month of data
        
        # Calculate maximum samples that fit in memory
        max_samples = int(self.max_memory_gb / sample_size_gb)
        
        # Ensure chunk size is multiple of batch size
        chunk_size = (max_samples // self.batch_size) * self.batch_size
        return min(chunk_size, self.num_samples)
    
    def _get_available_months(self) -> List[Tuple[int, int]]:
        """Get list of available months"""
        available_months = []
        for year in self.years:
            for month in self.months:
                if self.loader.is_data_available(year, month):
                    available_months.append((year, month))
        return available_months
    
    def _load_chunk(self, start_idx: int) -> None:
        """Load a chunk of data into memory"""
        end_idx = min(start_idx + self.chunk_size, self.num_samples)
        chunk_indices = self.indices[start_idx:end_idx]
        
        # Clear previous chunk to free memory
        self._data_cache.clear()
        
        # Determine optimal number of processes
        num_cores = cpu_count()
        num_processes = min(num_cores - 2, len(chunk_indices))  # Leave cores for system
        
        # Prepare data loading function
        def load_month_data(idx):
            year, month = self.available_months[idx]
            try:
                # Load data
                grace_data, grace_meta = self.loader.load_monthly_data(year, month)
                aux_data, aux_meta = self.loader.load_auxiliary_data(year, month)
                static_data, static_meta = self.loader.load_static_data()
                
                # Preprocess data
                grace_norm, aux_norm, static_norm = self.preprocessor.prepare_data(
                    grace_data, aux_data, static_data,
                    grace_meta, aux_meta, static_meta
                )
                
                return idx, (grace_norm, aux_norm, static_norm)
                
            except Exception as e:
                logger.error(f"Error loading data for {year}-{month:02d}: {e}")
                return None
        
        # Load data in parallel
        with Pool(num_processes) as pool:
            results = pool.map(load_month_data, chunk_indices)
        
        # Store valid results in cache
        for result in results:
            if result is not None:
                idx, data = result
                self._data_cache[idx] = data
    
    def __len__(self) -> int:
        """Get number of batches per epoch"""
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Get a batch of data"""
        start_idx = idx * self.batch_size
        
        # Check if we need to load new chunk
        chunk_start = (start_idx // self.chunk_size) * self.chunk_size
        if not self._data_cache or start_idx >= chunk_start + self.chunk_size:
            self._load_chunk(chunk_start)
        
        # Get batch indices
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Prepare batch data
        grace_batch = []
        aux_batch = []
        static_batch = []
        
        for idx in batch_indices:
            if idx in self._data_cache:
                grace_norm, aux_norm, static_norm = self._data_cache[idx]
                grace_batch.append(grace_norm)
                aux_batch.append(aux_norm)
                static_batch.append(static_norm)
        
        # Stack batch data
        grace_batch = np.stack(grace_batch)
        aux_batch = np.stack(aux_batch)
        static_batch = np.stack(static_batch)
        
        # Process for stage
        if self.stage == 2:
            # Adjust dimensions for stage 2
            target_batch = tf.image.resize(
                grace_batch,
                [grace_batch.shape[1] * 4, grace_batch.shape[2] * 4],
                method='bilinear'
            )
        else:
            # Stage 1 processing
            target_batch = tf.image.resize(
                grace_batch,
                [grace_batch.shape[1] * 2, grace_batch.shape[2] * 2],
                method='bilinear'
            )
        
        inputs = {
            'grace_input': grace_batch,
            'aux_input': aux_batch,
            'static_input': static_batch
        }
        
        return inputs, target_batch
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        # Clear cache to free memory
        self._data_cache.clear()

def test_generator():
    """Test the optimized generator"""
    try:
        # Create test generator
        gen = OptimizedGRACEGenerator(
            years=[2002, 2003],
            months=list(range(1, 13)),
            batch_size=32,
            stage=1,
            shuffle=True,
            max_memory_gb=16.0
        )
        
        # Test batch generation
        inputs, targets = gen[0]
        
        print("\nGenerator Test Results:")
        print(f"Number of samples: {len(gen)}")
        print("\nInput shapes:")
        for key, value in inputs.items():
            print(f"{key}: {value.shape}")
        print(f"Target shape: {targets.shape}")
        
        print("\nMemory usage:")
        process = psutil.Process()
        print(f"Current memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
        
        print("\nAll generator tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_generator()