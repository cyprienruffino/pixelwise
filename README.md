# Code for "Pixel-wise Conditioning of Generative Adversarial Networks", ESANN2019

## Usage

### Conditional gan
```python src/train_cgan.py config_file dataset_path```

### Regularized gan
```python src/train_rcgan.py config_file dataset_path```

### Regularized gan with PacGAN
```python src/train_rcgan_pac.py config_file dataset_path```

## Dataset formatting
Saved in HDF5 format. Use the code from scripts/datasets to format a dataset from a large single image or from classical datasets.

## Figures
The figures are generated from the scrips/plot scripts
