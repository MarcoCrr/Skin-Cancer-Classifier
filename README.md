# Skin Cancer Classification with PyTorch

<!-- [![codecov](https://codecov.io/github/MarcoCrr/Skin-Cancer-Classifier/graph/badge.svg?token=5INB1F11SK)](https://codecov.io/github/MarcoCrr/Skin-Cancer-Classifier) -->
[![Tests](https://github.com/MarcoCrr/Skin-Cancer-Classifier/actions/workflows/python-tests.yml/badge.svg)](https://github.com/MarcoCrr/Skin-Cancer-Classifier/actions)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end, PyTorch-based image classification pipeline for distinguishing **benign** and **malignant** skin lesions using transfer learning with **ResNet18** and the HAM10000 dataset.

The project emphasizes clean architecture and includes model evaluation, visualization, and testing. It also takes potential hardware memory constraints into account through configurable dataset size and data transformations. GPU training performance and data-loading efficiency were additionally benchmarked and investigated, as discussed in the [Training Performance Benchmark](#training-performance-benchmark) section.


## Features
### End-to-end ML pipeline:
* Data preparation
* Training
* Evaluation
* Visualization


### Evaluation:
* Precision / Recall
* Confusion Matrix
* ROC Curve
* Precision–Recall Curve
* Training-performance benchmarking and GPU utilization analysis

### Visualization tools:
* Predictions (with mistakes filtering)
* Training curves (loss & accuracy)


### Project Structure
```
.
├── configs/           # Configuration files
├── data/              # Dataset
├── logs/              # Outputs (plots, metrics)
├── models/            # Saved models
├── src/               # Source code
├── tests/             # Unit tests
├── README.md
```


## Examples: Results & Visualizations
**Notice**: this is a modest-sized project, whose goal is not to compete with more elaborate methods but rather to show how to set up a solid ML project, with good programming practices, et cetera. Its performance can be greatly improved with some tweaks and improvements.

### Confusion Matrix
![Confusion Matrix](logs/confusion_matrix.png)

### ROC Curve
![ROC Curve](logs/roc_curve.png)

### Precision-Recall Curve
![PR Curve](logs/precision_recall_curve.png)

### Predictions
![Predictions](logs/predictions.png)

### Training Curves
![Training](logs/training_curves.png)


### Installation
Clone repository
```
git clone https://github.com/MarcoCrr/Skin-Cancer-Classifier.git
cd Skin-Cancer-Classifier
```

### Create environment (recommended)
```
conda create -n torch_env python=3.10
conda activate torch_env
```

### Install dependencies
Inside the active environment:
```
pip install -r requirements.txt
```

### Dataset

This project uses the HAM10000 dataset (skin lesion images).

Download with Kaggle:
```
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/
```
... or manually from [this link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).


## Brief Tutorial
### Data Preparation

Prepare train/validation splits:

```
python -m src.prepare_data \
    --data_dir data \
    --output_dir data \
    --val_split 0.2 \
    --sample_size 2000
```
Training
```
python -m src.train --config configs/config.yaml
```

This will:

1. Train a ResNet18 model

2. Save the best model to _models/best_model.pth_

3. Log metrics to: _logs/train_log.txt_


### Evaluation
```
python -m src.evaluate
```

**Outputs:** precision / recall, confusion matrix, classification report

**Saved in:** _logs/eval.txt_


### Visualization
```
python -m src.visualize
```

Options:
```
--mistakes_only        # Show only incorrect predictions
--num_images           # Number of images to display
```


Generated plots saved in _logs/_:
```
training_curves.png
confusion_matrix.png
roc_curve.png
precision_recall_curve.png
predictions.png
```


### Testing

Run all tests:
```
pytest --cov=src
```

### Model Details
Architecture: ResNet18 <br>
Transfer learning (ImageNet pretrained) <br>
Final layer adapted for binary classification <br>


--------------------------------------


### Training Performance Benchmark
An additional part of the project investigates GPU training performance and data-loading efficiency. <br>
Running the benchmark:
```
python -m src.benchmark.py
```
It measures:

* images/second
* batch time
* total benchmark time
* data-loading time
* CPU to GPU transfer time
* forward-pass time
* backward-pass time
* optimizer time
* PyTorch GPU memory allocation
* CPU utilization
* GPU utilization
* total GPU memory usage

Output example:
```
    =======================================================
    PyTorch Training Benchmark
    =======================================================
    Date:               2026-09-01T11:10:38
    Session ID:         test
    Device:             cuda

    PyTorch:            2.5.1+cu121
    CUDA available:     True
    CUDA version:       12.1
    GPU:                NVIDIA GeForce RTX 2060
    Python:             3.10.20

    Configuration
    -------------------------------------------------------
    Batch Size:         32
    DataLoader Workers: 5
    Measured Batches:   50
    Images processed:   1600

    Performance
    -------------------------------------------------------
    Images / second:    797.63
    Batch time:         40.12 ms
    Total time:         2.01 s

    Timing breakdown
    -------------------------------------------------------
    Data loading:       1.38 ms
    CPU->GPU time:      10.02 ms
    Forward pass:       27.10 ms
    Backward pass:      0.67 ms
    Optimizer:          0.59 ms
    (Backw and opt tiny because I decided to freeze the backbone)

    GPU memory
    -------------------------------------------------------
    PyTorch peak allocated:   286.14 MB
    =======================================================
    
    System Monitoring
    -------------------------------------------------------
    CPU utilization
        Average:          39.31 %
        Maximum:          68.80 %

    GPU utilization
        Average:          51.56 %
        Maximum:          75.00 %

    GPU total memory used
        Average:          1123.58 MB
        Maximum:          1173.19 MB
    -------------------------------------------------------
```

#### Comments

Internal benchmarks showed that data loading is the main bottleneck when `num_workers=0`, with throughput around ~150–180 images/s. Increasing the number of workers produced a substantial improvement, reaching **800+ images/s**, with `num_workers=5` providing a good tradeoff for my setup. Batch size had a smaller impact on throughput; `batch_size=32` was selected as a good compromise between performance and GPU memory usage.

I then tested `pin_memory=True`, `persistent_workers=True`, and `non_blocking=True` individually and together. While these settings substantially reduced CPU-to-GPU transfer time, the measured data-loading wait increased by a similar amount, resulting in no meaningful throughput improvement.

I also evaluated **Automatic Mixed Precision (AMP)**. AMP reduced GPU forward-computation time by ~40% and PyTorch peak memory consumption by ~30%, while increasing backward and optimizer overhead. However, overall throughput remained essentially unchanged. The observed `data_time` increased by ~127%, suggesting that the faster GPU computation exposed more waiting for the next batch rather than making data transfer slower. GPU utilization also decreased, supporting this interpretation.

To investigate this further, I used the **PyTorch Profiler**. The profiling confirmed that the ~41% forward-time improvement comes from faster convolution, ReLU, and BatchNorm operations when using reduced-precision GPU kernels. Host-to-device transfer time remained essentially unchanged between FP32 and AMP, confirming that AMP does not significantly slow data transfer.

Overall, the benchmark identified `num_workers=5` and `batch_size=32` as the best parameters for my current setup. AMP successfully reduces GPU computation time and memory usage, but the current data pipeline prevents these improvements from translating into higher throughput.