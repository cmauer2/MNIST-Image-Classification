import os, argparse, random, json
from utils.data_utils import load_image_paths_by_class  # maps class -> list of paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='Path with subfolders 0..9 of images')   # MNIST-like layout
    ap.add_argument('--out_dir', required=True, help='Directory to save partitions')            # where to write jsons
    ap.add_argument('--train_frac', type=float, default=0.8)                                    # percentage for training
    ap.add_argument('--seed', type=int, default=1337)                                           # reproducible splits
    args = ap.parse_args()

    random.seed(args.seed)  # fix RNG for repeatability

    by_class = load_image_paths_by_class(args.data_root)
    train_idx = []
    val_idx = []

    for cls, paths in by_class.items():
        n = len(paths)  # number of images in class
        idx = list(range(n))    # indices
        random.shuffle(idx) # shuffle indices
        cut = int(args.train_frac * n)  # split point
        tr = idx[:cut]  # training indices
        va = idx[cut:]  # validation indices
        train_idx.extend([(paths[i], cls) for i in tr]) # collect (path, cls)
        val_idx.extend([(paths[i], cls) for i in va])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train.json'), 'w') as f:
        json.dump(train_idx, f)
    with open(os.path.join(args.out_dir, 'val.json'), 'w') as f:
        json.dump(val_idx, f)

    print(f"Saved train/val splits to {args.out_dir}. Train={len(train_idx)}, Val={len(val_idx)}")

if __name__ == '__main__':
    main()
