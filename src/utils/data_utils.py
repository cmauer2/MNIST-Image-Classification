import os
import glob
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict

def load_image_paths_by_class(data_root: str) -> Dict[int, List[str]]:
    """Assumes data_root/<digit>/*.png (or jpg). Returns {class: [paths...]}."""
    paths = {}
    for cls in range(10):
        cls_dir = os.path.join(data_root, str(cls))         # digits 0..9
        exts = ('*.png','*.jpg','*.jpeg','*.bmp')           # subfolder per class & supported types
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(cls_dir, e)))       # collect files
        files.sort()
        paths[cls] = files
    return paths

def read_grayscale(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')  # 0..255
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize [0,1]
    return arr

def flatten_images(imgs: np.ndarray) -> np.ndarray:
    # imgs: (N, H, W) -> (N, H*W)
    return imgs.reshape(imgs.shape[0], -1)      # flatten per image

def make_dataset(data_root: str, index_list: List[Tuple[str,int]], keep_2d: bool=False):
    """index_list: list of (path, label). Returns (X, y).
    If keep_2d: X shape is (N,1,28,28); else (N, 784).
    """
    X = []
    y = []
    for p, lab in index_list:       # 
        im = read_grayscale(p)      # read and normalize one image
        if keep_2d:
            X.append(im[None, ...])  # add channel dim -> (1, H, W)
        else:
            X.append(im.reshape(-1))    # flatten to vector
        y.append(lab)   # store label
    X = np.stack(X) # stack into array
    y = np.array(y, dtype=np.int64) # labels as int64
    return X, y

def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())        # fraction correct

def one_hot(y: np.ndarray, num_classes: int=10) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)  # zeroes
    oh[np.arange(y.shape[0]), y] = 1.0      # set 1s
    return oh
