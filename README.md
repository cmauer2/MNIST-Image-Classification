
# MNIST Pattern Recognition Project

**Implements:** KNN (NumPy), Naïve Bayes (NumPy), Linear Classifier (NumPy or PyTorch), MLP (PyTorch), CNN (PyTorch).  
**Dataset:** MNIST raw images organized as `data_root/<digit>/*.png` (or jpg).

## Quick Start

1) Put the *raw MNIST image folders (0..9)* from Moodle under `./data/mnist_raw/0`, `./data/mnist_raw/1`, ..., `./data/mnist_raw/9`.

2) Create train/val partitions **per your own split** (default: 80/20):
```
python src/utils/make_partitions.py --data_root data/mnist_raw --out_dir partitions --train_frac 0.8 --seed 1337
```

3) Run KNN (NumPy-only):
```
python src/numpy_impl/knn_numpy.py --data_root data/mnist_raw --partitions partitions --k 3
```

4) Run Naïve Bayes (NumPy-only, with binarization at 0.5):
```
python src/numpy_impl/naive_bayes_numpy.py --data_root data/mnist_raw --partitions partitions
```

5) Run Linear (choose NumPy or PyTorch variant):
```
# NumPy
python src/numpy_impl/linear_numpy.py --data_root data/mnist_raw --partitions partitions --lr 0.1 --epochs 10 --l2 0.0

# PyTorch
python src/torch_impl/linear_torch.py --data_root data/mnist_raw --partitions partitions --lr 0.1 --epochs 10
```

6) Run MLP (PyTorch):
```
python src/torch_impl/mlp_torch.py --data_root data/mnist_raw --partitions partitions --lr 0.1 --epochs 10 --batch_size 128
```

7) Run CNN (PyTorch):
```
python src/torch_impl/cnn_torch.py --data_root data/mnist_raw --partitions partitions --lr 0.01 --epochs 10 --batch_size 128
```

8) Visualizations & analysis:
```
python src/utils/eval_and_viz.py --data_root data/mnist_raw --partitions partitions --confusion --show_examples --save_dir figures
```

## Notes
- KNN & Naïve Bayes are **NumPy only** as required.
- For KNN we compute **Euclidean distance** and use majority vote, with ties broken by smallest index.
- Naïve Bayes uses Laplace smoothing.
- Linear classifier trains with **L2 loss** (MSE) against one-hot labels, using gradient descent.
- MLP & CNN train with cross-entropy (standard in classification).

See `reports/report_template.md` for your write-up structure.
