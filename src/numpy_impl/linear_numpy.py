import os, argparse, json
import numpy as np
from utils.data_utils import make_dataset, one_hot

# Function: mean squared error (MSE) loss

def mse_loss(pred, target):
    return ((pred - target)**2).mean()      #calculates average of squaared differences b/w predictions and true labels

# Main training script for the Linear Classifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--partitions', required=True)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--l2', type=float, default=0.0, help='weight decay')   # optional L2 regularization (prevents overfitting)
    args = ap.parse_args()

    # Load partition JSON files that list image paths and labels
    with open(os.path.join(args.partitions, 'train.json'),'r') as f:
        train_idx = json.load(f)
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)

    # Load actual image data as numpy array
    Xtr, ytr = make_dataset(args.data_root, train_idx, keep_2d=False)   # training data
    Xva, yva = make_dataset(args.data_root, val_idx, keep_2d=False)     # validation data

    # Get dataset dimensions
    N, D = Xtr.shape        # N = # of training samples, D = features per sample 
    C = 10                  # # of output classes
    
    # Initialize weights with small random numbers
    W = np.random.randn(D, C).astype(np.float32) * 0.01

    # Convert labels into one hot encoded format
    ytr_oh = one_hot(ytr, C)

    # Training Loop (gradient descent)
    for ep in range(args.epochs):
        # forward
        logits = Xtr @ W  # (N,C)
        
        # compute loss: MSE b/w predictions and true one hot labels + L2 regularization
        loss = mse_loss(logits, ytr_oh) + args.l2 * (W*W).sum()
        # gradient wrt W
        grad = (2.0 / (N)) * (Xtr.T @ (logits - ytr_oh)) + 2*args.l2*W
        # update
        W -= args.lr * grad

        # Validation phase (check the accuracy on unseen data)
        val_logits = Xva @ W
        preds = np.argmax(val_logits, axis=1)
        acc = (preds == yva).mean()
        print(f"Epoch {ep+1}/{args.epochs} - loss {loss:.4f} - val acc {acc:.4f}")

    # Save trained weights and validation predicts
    os.makedirs('experiments', exist_ok=True)
    np.save('experiments/linear_numpy_W.npy', W)
    np.save('experiments/linear_numpy_val_preds.npy', preds)
    print('Saved W and predictions to experiments/.')

if __name__ == '__main__':
    main()
