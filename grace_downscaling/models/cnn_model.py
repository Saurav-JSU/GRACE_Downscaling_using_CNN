import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
from typing import Tuple, Dict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualBlock(layers.Layer):
    """Residual block with batch normalization"""
    
    def __init__(self, filters: int, kernel_size: int = 3):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Projection shortcut if needed
        self.projection = layers.Conv2D(filters, 1, padding='same')
        
    def call(self, inputs):
        shortcut = self.projection(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = layers.add([shortcut, x])
        return tf.nn.relu(x)

class GRACEDownscalingModel:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        # Enable mixed precision for speed
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    @staticmethod
    def _conv_block(x, filters, kernel_size=3):
        """Fast convolution block"""
        x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)
    
    def build_stage1_model(self):
        # Input layers with correct shape and dtype
        grace_input = layers.Input(shape=(1, 9, 17), dtype='float32', name='grace_input')
        aux_input = layers.Input(shape=(28, 9, 17), dtype='float32', name='aux_input')
        static_input = layers.Input(shape=(6, 9, 17), dtype='float32', name='static_input')
        
        # Efficient processing branches
        x_grace = self._conv_block(grace_input, 64)
        x_aux = self._conv_block(aux_input, 64)
        x_static = self._conv_block(static_input, 64)
        
        # Fast concatenation
        x = layers.Concatenate(axis=1)([x_grace, x_aux, x_static])
        
        # Deeper network since we have memory
        for _ in range(4):  # More layers for better accuracy
            x = self._conv_block(x, 192)  # Wider layers
        
        # Upsampling with transposed convolution
        x = layers.Conv2DTranspose(96, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Final convolution
        outputs = layers.Conv2D(1, 1)(x)
        outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs)
        
        model = Model([grace_input, aux_input, static_input], outputs)
        
        # Use AdamW optimizer with weight decay
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01
        )
        
        # Compile with mixed precision
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
            jit_compile=True  # Enable XLA compilation
        )
        
        return model
    
    def build_stage2_model(self):
        # Similar to stage 1 but with adjusted dimensions
        grace_input = layers.Input(shape=(1, 18, 34), dtype='float32', name='grace_input')
        aux_input = layers.Input(shape=(28, 18, 34), dtype='float32', name='aux_input')
        static_input = layers.Input(shape=(6, 18, 34), dtype='float32', name='static_input')
        
        x_grace = self._conv_block(grace_input, 128)  # Wider network for stage 2
        x_aux = self._conv_block(aux_input, 128)
        x_static = self._conv_block(static_input, 128)
        
        x = layers.Concatenate(axis=1)([x_grace, x_aux, x_static])
        
        # Deeper network for stage 2
        for _ in range(6):
            x = self._conv_block(x, 384)
        
        # Two upsampling steps
        x = layers.Conv2DTranspose(192, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(96, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv2D(1, 1)(x)
        outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs)
        
        model = Model([grace_input, aux_input, static_input], outputs)
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
            jit_compile=True
        )
        
        return model

def test_model():
    """Test function to verify model creation and basic operations"""
    try:
        # Initialize model
        grace_model = GRACEDownscalingModel()
        
        # Test Stage 1 model
        print("\nTesting Stage 1 Model:")
        stage1_model = grace_model.build_stage1_model()
        stage1_model.summary()
        
        # Test with dummy data
        batch_size = 4
        grace_data = tf.random.normal((batch_size, 9, 17, 1))
        aux_data = tf.random.normal((batch_size, 9, 17, 28))
        static_data = tf.random.normal((batch_size, 9, 17, 6))
        
        stage1_output = stage1_model([grace_data, aux_data, static_data])
        print(f"\nStage 1 output shape: {stage1_output.shape}")
        
        # Test Stage 2 model
        print("\nTesting Stage 2 Model:")
        stage2_model = grace_model.build_stage2_model()
        stage2_model.summary()
        
        # Test with stage 1 output shape
        grace_data_stage2 = tf.random.normal((batch_size, 18, 34, 1))
        aux_data_stage2 = tf.random.normal((batch_size, 18, 34, 28))
        static_data_stage2 = tf.random.normal((batch_size, 18, 34, 6))
        
        stage2_output = stage2_model([grace_data_stage2, aux_data_stage2, static_data_stage2])
        print(f"\nStage 2 output shape: {stage2_output.shape}")
        
        print("\nAll model tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_model()