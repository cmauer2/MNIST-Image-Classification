import os, argparse, json
import numpy as np
from collections import Counter
from utils.data_utils import make_dataset

# Function: Compute Euclidean distance between test and train

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a has shape (N, D) = test samples, b has (M, D) = train samples
    # using math trick -> (a-b)^2 = a^2 + b^2 -2ab for efficiency
    a2 = (a**2).sum(axis=1, keepdims=True)  # (N,1) -> sum of squares for each test sample
    b2 = (b**2).sum(axis=1, keepdims=True).T  # (1,M) -> sum of squares for each train sample
    ab = a @ b.T  # (N,M) -> matrix multiply gives dot products
    d2 = a2 + b2 - 2*ab
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)

# Function: predict labels using K nearest Neighbors

def predict_knn(X_train, y_train, X_test, k=3):
    dists = euclidean_distance(X_test, X_train)
    idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]  # (N,k) -> get indexes of k closest points
    preds = []
    for i in range(idx.shape[0]):                         # loop over each image
        labs = y_train[idx[i]]
        c = Counter(labs.tolist())
        # tie-break: smallest class index
        max_count = max(c.values())
        candidates = [cls for cls, ct in c.items() if ct == max_count] 
        preds.append(min(candidates))
    return np.array(preds, dtype=np.int64)  # return predicitons as NumPy array

# Main program: loads data, runs KNN, saves predictions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--partitions', required=True)
    ap.add_argument('--k', type=int, default=3)
    args = ap.parse_args()

    # Load the JSON index files for training and validation sets
    with open(os.path.join(args.partitions, 'train.json'),'r') as f:
        train_idx = json.load(f)
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)

    # Load and flatten the actual image data into numpy arrays
    Xtr, ytr = make_dataset(args.data_root, train_idx, keep_2d=False)   # training data
    Xva, yva = make_dataset(args.data_root, val_idx, keep_2d=False)     # validation data

    preds = predict_knn(Xtr, ytr, Xva, k=args.k)
    acc = (preds == yva).mean()
    print(f"KNN (k={args.k}) Val accuracy: {acc:.4f}")
    out = os.path.join('experiments', f'knn_k{args.k}_val_preds.npy')
    os.makedirs('experiments', exist_ok=True)
    np.save(out, preds)
    print(f"Saved val predictions to {out}")

if __name__ == '__main__':
    main()
