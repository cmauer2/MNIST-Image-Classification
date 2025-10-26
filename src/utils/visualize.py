import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_linear_weights(W_path: str, out_dir='figures'):
    W = np.load(W_path)  # load (784,10) weight matrix
    os.makedirs(out_dir, exist_ok=True) # this makes output folder if missing
    plt.figure()        
    for c in range(10):
        plt.subplot(2,5,c+1)        # create figure, one subplot per class, 2 rows x 5 col grid
        plt.imshow(W[:,c].reshape(28,28))
        plt.axis('off')
        plt.title(f'W class {c}')
    plt.savefig(os.path.join(out_dir, 'linear_W_filters.png'), bbox_inches='tight')     # save figure
    print(f"Saved linear weight visualizations to {out_dir}/linear_W_filters.png")

def visualize_nb_theta(theta_path: str, out_dir='figures'):
    theta = np.load(theta_path)  # load (10,784) Naive Bayes theta
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for c in range(10):
        plt.subplot(2,5,c+1)
        plt.imshow(theta[c].reshape(28,28))     # show per class prob map
        plt.axis('off')
        plt.title(f'theta class {c}')
    plt.savefig(os.path.join(out_dir, 'nb_theta_maps.png'), bbox_inches='tight')
    print(f"Saved Naive Bayes theta visualizations to {out_dir}/nb_theta_maps.png")
