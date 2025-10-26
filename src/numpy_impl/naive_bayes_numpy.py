import os, argparse, json
import numpy as np
from utils.data_utils import make_dataset

# Function to binarize pixel values (turn them into 0 or 1)
def binarize(X: np.ndarray, threshold: float=0.5) -> np.ndarray:
    return (X >= threshold).astype(np.float32)

# Function to train the Naive Bayes model
def fit_bernoulli_nb(X: np.ndarray, y: np.ndarray, num_classes=10, alpha=1.0):
    # X: (N, 784) binary, y: (N,)
    N, D = X.shape
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float64)  # (C,) -> count how many samples per class
    log_prior = np.log(class_counts / N + 1e-12)  # (C,) -> calculate log of prior probabilites

    # Conditional probs with Laplace smoothing
    # theta[c, d] = P(x_d=1 | y=c)
    theta = np.zeros((num_classes, D), dtype=np.float64)
    for c in range(num_classes):    # loop through each digit (0-9)
        Xc = X[y==c]
        # sum over samples + alpha / (Nc + 2alpha)
        Nc = max(1, Xc.shape[0])
        # apply laplace smoothing: add alpha to numerator and 2*alpha to denominator
        theta[c] = (Xc.sum(axis=0) + alpha) / (Nc + 2*alpha)

    return log_prior, theta

# Function to predict class labels for new data
def predict_bernoulli_nb(X: np.ndarray, log_prior, theta):
    # for each sample, compute log using the Bernoulli formula
    log_theta = np.log(theta + 1e-12)       # log of theta
    log_one_minus = np.log(1 - theta + 1e-12)   # log of (1 - theta)
    # X: (N,D) -> compute log-probabilites for each class
    lp = X @ log_theta.T + (1 - X) @ log_one_minus.T  # (N,C) -> likelihood for each sample and class
    scores = lp + log_prior[None, :]    #add prior to get final score
    return np.argmax(scores, axis=1)    # pick the class with max log-probability

# Main function to tie everything together
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--partitions', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=1.0)
    args = ap.parse_args()

    # load the partition files that list training and validation data
    with open(os.path.join(args.partitions, 'train.json'),'r') as f:
        train_idx = json.load(f)
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)

    # load actual image data and labels into numpy arrays
    Xtr, ytr = make_dataset(args.data_root, train_idx, keep_2d=False)
    Xva, yva = make_dataset(args.data_root, val_idx, keep_2d=False)

    # binarize both training and validation sets
    Xtrb = binarize(Xtr, args.threshold)
    Xvab = binarize(Xva, args.threshold)

    # train Naive Bayes model and then predict on validation data
    log_prior, theta = fit_bernoulli_nb(Xtrb, ytr, num_classes=10, alpha=args.alpha)
    preds = predict_bernoulli_nb(Xvab, log_prior, theta)
    
    # check how many predictions match true labels
    acc = (preds == yva).mean()
    print(f"Naïve Bayes (threshold={args.threshold}, alpha={args.alpha}) Val accuracy: {acc:.4f}")
    
    # save results to visualize them later
    os.makedirs('experiments', exist_ok=True)
    np.save('experiments/nb_val_preds.npy', preds)
    np.save('experiments/nb_theta.npy', theta)
    print("Saved predictions and theta to experiments/.")

if __name__ == '__main__':
    main()
