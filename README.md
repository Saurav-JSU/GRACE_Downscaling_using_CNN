# GRACE Downscaling Project

This `README.md` file provides an overview of the project, its structure, setup instructions, and usage guidelines.

This project focuses on downscaling GRACE (Gravity Recovery and Climate Experiment) data using various auxiliary datasets. The project is structured to preprocess data, train models, and evaluate the results.

## Project Structure
. ├── a.py ├── Getting GEE data.ipynb ├── grace_downscaling/ │ ├── init.py │ ├── data/ │ │ ├── init.py │ │ ├── data_diagnostics.py │ │ ├── data_generator.py │ │ ├── data_loader.py │ │ ├── data_validator.py │ │ └── preprocessor.py │ ├── models/ │ │ └── cnn_model.py │ ├── training/ │ │ └── train.py │ └── utils/ │ └── init.py ├── grace_downscaling.egg-info │ ├── dependency_links.txt │ ├── PKG-INFO │ ├── requires.txt │ ├── SOURCES.txt │ └── top_level.txt ├── Processing GEE data.ipynb ├── README.md ├── requirements.txt ├── setup.py └── tests/

## Notebooks

- **Getting GEE data.ipynb**: This notebook is used to collect data from Google Earth Engine (GEE). It processes various datasets such as MODIS, GLDAS, and ERA5.
- **Processing GEE data.ipynb**: This notebook processes the collected GEE data, prepares it for model training, and exports it to Google Drive.

## Modules

### Data

- **data_diagnostics.py**: Contains functions for diagnosing data issues.
- **data_generator.py**: Generates data for training and validation.
- **data_loader.py**: Loads data from various sources.
- **data_validator.py**: Validates the integrity and quality of the data.
- **preprocessor.py**: Preprocesses the data, including resampling and normalization.

### Models

- **cnn_model.py**: Defines the Convolutional Neural Network (CNN) model used for downscaling.

### Training

- **train.py**: Contains the main training loop and functions for training the model.

### Utils

- **config.py**: Configuration settings for the project.

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up the project:
    ```sh
    python setup.py install
    ```

## Usage

### Data Collection

Run the [Getting GEE data.ipynb](http://_vscodecontentref_/8) notebook to collect data from Google Earth Engine.

### Data Processing

Run the [Processing GEE data.ipynb](http://_vscodecontentref_/9) notebook to process the collected data and prepare it for model training.

### Training

Run the training script:
```sh
python grace_downscaling/training/train.py
