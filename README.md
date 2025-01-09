# Explainable AI (XAI) Project

This project implements various XAI techniques to explain predictions made by deep learning models for both image classification (tumor detection) and text analysis (tweet sentiment).

## Features

- Image Classification XAI:
  - GradCAM
  - SHAP
  - LIME
- Text Analysis XAI:
  - SHAP
  - LIME
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
├── utils/                    # Utility functions
├── xai/                     # XAI implementations
├── main.py                  # Main execution script
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- transformers >= 4.11.0
- SHAP >= 0.40.0
- LIME >= 0.2.0
- scikit-image >= 0.18.0
- matplotlib >= 3.4.0
- numpy >= 1.19.0
- pandas >= 1.3.0

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

The main script now supports modular execution through command line arguments:

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
- Image explanations: `explanations_[i].png`
- Text explanations: `text_explanations_[i].png`

Each visualization includes the original input and explanations from different XAI methods. The script also generates an execution log file that tracks the processing of each sample.

## TODO

Currently only image part works. 

- [ ] Verify image part output. 
- [ ] Fix text part.
