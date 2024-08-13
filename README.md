# Trigger Word Detection Project

## Overview

This project implements a deep learning model to detect the word 'boy' in MP3 files. It utilizes Azure Machine Learning for training, evaluation, and deployment of the model.

## Project Structure

The project consists of three main pipelines:

1. **Training and Evaluation Pipeline** (`run_pipeline.py`)
2. **Model Deployment Pipeline** (`run_deployment.py`)
3. **Inference Pipeline** (using Streamlit - `run_streamlit.py`)

### Key Components

- `load_data.py`: Loads audio samples from Azure Blob Storage
- `extract_features.py`: Extracts MFCC features from audio files
- `prepare_dataset.py`: Prepares features for model training
- `split_data.py`: Splits dataset into train, validation, and test sets
- `training.py`: Trains the deep learning model (CNN-GRU architecture)
- `evaluation.py`: Evaluates the trained model
- `register_model.py`: Registers the model in Azure ML
- `deploy_model.py`: Deploys the model as an online endpoint

## Setup and Installation

1. Ensure you have an Azure account with access to Azure Machine Learning.
2. Install the required Python packages:

   ```
   pip install azure-ai-ml azure-identity mlflow tensorflow
   ```

3. Set up your Azure ML workspace and configure your credentials.

## Usage

### Training and Evaluation

Run the training pipeline:

```
python run_pipeline.py
```

This script will:

- Load and process the data
- Train the model
- Evaluate its performance
- Register the model if it meets the accuracy threshold

### Model Deployment

Deploy the trained model:

```
python run_deployment.py
```

This will create or update an online endpoint with the latest version of your model.

### Inference

(Note: Implementation details for the Streamlit app are not provided in the current codebase)

Run the Streamlit app for inference:

```
streamlit run run_streamlit.py
```

This should launch a web interface where you can upload MP3 files for trigger word detection.

## Technical Details

- **Model Architecture**: CNN-GRU
- **Feature Extraction**: MFCC (Mel-frequency cepstral coefficients)
- **Training Parameters**:
  - Epochs: 50
  - Batch Size: 32
  - Optimizer: Adam
  - Loss: Binary Crossentropy

## Azure ML Integration

- Uses Azure ML for experiment tracking and model management
- Implements MLflow for metric logging
- Utilizes Azure Key Vault for secure storage of sensitive information

## Future Improvements

1. Implement real-time audio processing
2. Explore advanced model architectures (e.g., Transformers)
3. Add support for multiple trigger words or languages
4. Implement A/B testing for deployed models

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
