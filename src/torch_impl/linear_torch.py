import os, argparse, json, torch, numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# dataset that flattens MNIST images to 784-d vectors
class FlatMNIST(Dataset):
    def __init__(self, index_list, normalize_to_neg1=False):
        self.index_list = index_list
        if normalize_to_neg1:
            self.t = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,),(0.5,))])
        else:
            self.t = transforms.ToTensor()

    def __len__(self): return len(self.index_list)      # dataset size

    def __getitem__(self, idx):
        path, lab = self.index_list[idx]
        x = Image.open(path).convert('L')
        x = self.t(x).view(-1)  # flatten to 784
        return x, lab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=False, help='unused, paths come from partitions jsons')
    ap.add_argument('--partitions', required=True)      # train/val json folder
    ap.add_argument('--lr', type=float, default=0.1)    #learning rate
    ap.add_argument('--epochs', type=int, default=10)   # epochs
    ap.add_argument('--batch_size', type=int, default=256)  # batch size
    ap.add_argument('--normalize_to_neg1', action='store_true')
    args = ap.parse_args()

    # read index files
    with open(os.path.join(args.partitions, 'train.json'),'r') as f:
        train_idx = json.load(f)
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)

    # dataset + loaders
    train_ds = FlatMNIST(train_idx, normalize_to_neg1=args.normalize_to_neg1)
    val_ds   = FlatMNIST(val_idx, normalize_to_neg1=args.normalize_to_neg1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)   #shuffle
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)                   # no shuffle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #pick device

    model = nn.Linear(784, 10).to(device)       # single linear layer
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)   # optimizer
    loss_fn = nn.CrossEntropyLoss()                     # loss for logits

    # train
    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # evaluate
        model.eval()
        correct = total = 0
        preds_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), torch.tensor(yb).to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)            # predicted class
                preds_all.append(preds.cpu().numpy())   # stash preds
                correct += (preds == yb).sum().item()   # correct count
                total += yb.size(0)                     # total samples

        acc = correct/total # accuracy
        print(f"Epoch {ep+1}/{args.epochs} - val acc {acc:.4f}")

    os.makedirs('experiments', exist_ok=True)
    torch.save(model.state_dict(), 'experiments/linear_torch.pt')
    np.save('experiments/linear_torch_val_preds.npy', np.concatenate(preds_all))
    print("Saved model and predictions to experiments/.")

if __name__ == '__main__':
    main()
