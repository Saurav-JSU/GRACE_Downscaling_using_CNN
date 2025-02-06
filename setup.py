from setuptools import setup, find_packages

setup(
    name="grace_downscaling",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'rasterio>=1.2.0',
        'scikit-learn>=0.24.0',
        'tensorflow>=2.8.0',
        'matplotlib>=3.4.0'
    ],
    python_requires='>=3.8'
)