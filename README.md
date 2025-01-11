# Explainable AI (XAI) Project

This project implements various XAI techniques to explain predictions made by deep learning models for both image classification (tumor detection) and text analysis (tweet sentiment).

## Features

- Image Classification:
  - Transfer Learning with ResNet50
  - K-fold Cross-validation Training
  - Data Augmentation Pipeline
  - Optimized GPU Training
  - Model Performance: 100% Validation Accuracy
- Image Classification XAI:
  - GradCAM
  - SHAP
  - LIME
- Text Analysis XAI:
  - SHAP
  - LIME
- Quantitative Evaluation:
  - Faithfulness scoring
  - Method comparison
  - Performance recommendations
- Modular execution for separate testing of image and text components

## Project Structure

```
.
├── data/
│   ├── tumor_detection/      # Image dataset
│   │   ├── pos/             # Malignant tumor images
│   │   └── neg/             # Benign tumor images
│   └── tweet_sentiment_extraction/  # Text dataset
├── models/                   # Neural network models
│   ├── image_model.py       # Image classification model
│   ├── text_model.py        # Text sentiment model
│   └── train_image_model.py # Image model training pipeline
├── utils/                    # Utility functions
│   ├── data_loader.py       # Dataset loading utilities
│   ├── evaluator.py         # XAI evaluation metrics
│   └── visualizer.py        # Visualization utilities
├── xai/                     # XAI implementations
│   ├── gradcam.py          # GradCAM implementation
│   ├── lime_explainer.py   # LIME wrapper
│   └── shap_explainer.py   # SHAP wrapper
├── main.py                  # Main execution script
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0 (with CUDA support)
- torchvision >= 0.10.0
- CUDA toolkit (optional, for GPU acceleration)
- transformers >= 4.11.0
- SHAP >= 0.40.0
- LIME >= 0.2.0
- scikit-learn >= 1.6.0
- scikit-image >= 0.18.0
- matplotlib >= 3.4.0
- numpy >= 1.19.0
- pandas >= 1.3.0
- tqdm >= 4.65.0
- Pillow >= 10.0.0

## Hardware Acceleration

The project now supports CUDA acceleration for improved performance on NVIDIA GPUs. Previously limited to CPU execution on M1 Macs, the implementation now automatically detects and utilizes available CUDA devices for:
- Model inference
- XAI computations (GradCAM, SHAP, LIME)
- Batch processing

No code changes are required - the system will automatically use GPU acceleration when available.

120 seconds (On M1 CPU) vs 5 seconds (RTX 4090 GPU) for a single image.  

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd XAI
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Tumor Detection Images
Place your tumor detection images in the following structure:
```
data/tumor_detection/
├── pos/          # Malignant tumor images
└── neg/          # Benign tumor images
```

### Tweet Sentiment Data
Place your tweet sentiment CSV files in:
```
data/tweet_sentiment_extraction/
└── train.csv     # Tweet sentiment training data
```

## Usage

### Model Training

Train the tumor detection model with optimized settings:
```bash
python models/train_image_model.py
```

This will:
- Implement 5-fold cross-validation
- Use transfer learning with ResNet50
- Apply data augmentation
- Optimize GPU utilization
- Save the best model weights

Current model performance:
- Prediction Confidence: 1.000 (improved from 0.522)
- Validation Accuracy: 100% ± 0% across all folds
- Training Time: ~15 epochs per fold

### Inference and XAI

The main script supports modular execution through command line arguments:

```bash
# Process both image and text datasets (default)
python main.py

# Process only tumor detection images
python main.py --mode image

# Process only tweet sentiment analysis
python main.py --mode text

# Process a specific number of samples
python main.py --mode all --samples 5
```

### Command Line Arguments

- `--mode`: Choose which dataset to process
  - `all`: Process both image and text datasets (default)
  - `image`: Process only tumor detection images
  - `text`: Process only tweet sentiment analysis
- `--samples`: Number of samples to process (default: 3)

For help with command line arguments:
```bash
python main.py --help
```

## Output

The results will be saved in a timestamped directory under `results/`. For each processed sample:

### Visualizations
- Image explanations: `explanations_[i].png`
  - Original image
  - GradCAM heatmap
  - SHAP values
  - LIME regions
- Text explanations: `text_explanations_[i].png`
  - SHAP token importance
  - LIME word contributions

### Quantitative Evaluation
The script generates an `execution.log` file containing:
- Model predictions
- Explanation method evaluations:
  - Faithfulness scores
  - Value ranges
  - Method comparisons
- Performance recommendations

### Evaluation Metrics

1. **Faithfulness**: Measures how well explanations reflect model decisions
   - Higher scores indicate better alignment with model behavior
   - Calculated by measuring prediction changes when important features are removed

2. **Method Comparison**: Analyzes relative performance of XAI methods
   - Identifies best performing method
   - Generates improvement recommendations
   - Provides value ranges and statistical measures

## Evaluation Results Example

```
XAI Method Evaluation:
GradCAM - Faithfulness: 0.052, Range: 0.000 to 0.992
SHAP - Faithfulness: 0.012, Range: 0.001 to 0.006
LIME - Faithfulness: 0.006, Range: -2.118 to 2.640

Method Comparison:
Best performing method: GradCAM
Recommendations:
- Consider improving feature attribution if faithfulness scores are low
- Analyze consistency across similar inputs
```

## TODO


- [ ] Verify output quality for both sections. Make optimizations. 

- [ ] Create presentation.
