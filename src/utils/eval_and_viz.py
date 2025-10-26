import os, argparse, json
import numpy as np
from sklearn.metrics import confusion_matrix    # to compute confusion matrix
import matplotlib.pyplot as plt
from utils.data_utils import make_dataset   # loads arrays

def plot_confusion(cm, classes, save_path=None):
    plt.figure()    # new figure
    plt.imshow(cm, interpolation='nearest')     # show matrix as image
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')     # save if path given
    else:
        plt.show()  

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)   # images root
    ap.add_argument('--partitions', required=True)  # val json path
    ap.add_argument('--preds', help='Path to .npy predictions for val split')   # model preds
    ap.add_argument('--confusion', action='store_true') # plot confusion?
    ap.add_argument('--show_examples', action='store_true') #plot misclassifications
    ap.add_argument('--save_dir', default='figures')    # where to save figures
    args = ap.parse_args()

    # load validation features/labels
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)
    Xv, yv = make_dataset(args.data_root, val_idx, keep_2d=False)

    if args.preds is None:
        print("Provide --preds=<val_preds.npy> from a model to analyze.")   # early exit
        return

    preds = np.load(args.preds)
    acc = (preds == yv).mean()
    print(f"Val accuracy: {acc:.4f}")

    if args.confusion:
        cm = confusion_matrix(yv, preds, labels=list(range(10)))    # 10 x 10 matrix
        os.makedirs(args.save_dir, exist_ok=True)
        plot_confusion(cm, list(range(10)), save_path=os.path.join(args.save_dir,'confusion.png'))

    if args.show_examples:
        # Show a grid of misclassified examples
        wrong = np.where(preds != yv)[0][:36]
        if len(wrong) == 0:
            print("No misclassifications to show.")     # nothing to ploy
            return
        W = Xv[wrong].reshape(len(wrong),28,28)         # back to image shape
        plt.figure()
        for i in range(len(wrong)):
            plt.subplot(6,6,i+1)                # 6x6 grid
            plt.imshow(W[i], cmap='gray')
            plt.axis('off')
            plt.title(f"t={yv[wrong][i]}, p={preds[wrong][i]}", fontsize=8)     # true/pred
        os.makedirs(args.save_dir, exist_ok=True)
        plt.savefig(os.path.join(args.save_dir,'misclassified.png'), bbox_inches='tight')

if __name__ == '__main__':
    main()
