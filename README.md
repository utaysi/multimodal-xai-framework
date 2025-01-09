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

Run the main script to generate XAI explanations:

```bash
python main.py
```

The script will:
1. Load both image and text models
2. Process 3 sample images and 3 sample tweets
3. Generate explanations using different XAI techniques
4. Save visualizations in the `results/xai_results_[timestamp]/` directory

## Output

The results will be saved in a timestamped directory under `results/`. For each processed sample:
- Image explanations: `explanations_[i].png`
- Text explanations: `text_explanations_[i].png`

Each visualization includes the original input and explanations from different XAI methods.

## TODO

Currently only image part works. 

- [ ] Verify image part output. 

- [ ] Fix text part.
