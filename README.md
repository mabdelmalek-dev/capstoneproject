# Check Fraud Detection System

This repository includes two key components for detecting fraudulent bank checks using OCR and Deep Learning:

![Demo](./capstoneProject.png)

## Project Structure

- **OCR_Module/**: Extracts numeric amounts from scanned check images using EasyOCR and compares them with stored values for validation.
- **ML_Module/**: Trains a convolutional neural network (CNN) to classify checks as fraudulent or legitimate based on image features.

## Requirements

- Python 3.x
- OpenCV
- EasyOCR
- TensorFlow
- NumPy
- Matplotlib
- PIL (Pillow)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## How It Works

### OCR Module
1. Crops the amount box from check images.
2. Uses EasyOCR to recognize numeric values.
3. Compares OCR result with CSV ground-truth and evaluates with confusion matrix.

### ML Module
1. Preprocesses and crops check image regions.
2. Trains a CNN model on labeled fraud/non-fraud data.
3. Evaluates model accuracy, precision, recall.
4. Uses the trained model to classify new check images.

## Dataset Structure

```
dataset/
  OCR/
    amount box/
    fraud-nonfraud/
    Check Amounts.csv
  ML/
    data/
      fraud/
      non fraud/
    uncropped data/
      fraud/
      non fraud/
    unscanned checks/
    unscanned uncropped checks/
    model output/
      fraud/
      non fraud/
```

## Author
Prepared for academic and practical demonstration of document fraud detection using AI tools.

