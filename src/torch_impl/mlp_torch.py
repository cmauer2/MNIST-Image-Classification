import os, argparse, json, torch, numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# dataset: flatten each image to 784-d vector in [0,1]
class FlatMNIST(Dataset):
    def __init__(self, index_list):
        self.index_list = index_list
        self.t = transforms.ToTensor()  # # converts to [0,1] tensor

    def __len__(self): return len(self.index_list)

    def __getitem__(self, idx):
        path, lab = self.index_list[idx]
        x = Image.open(path).convert('L')       # read grayscale
        x = self.t(x).view(-1)  # flatten to 784
        return x, lab

# Simple MLP: 784 -> 256 -> 128 -> 10
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),        # hidden layer 1
            nn.ReLU(),
            nn.Linear(256, 128),        # hidden layer 2
            nn.ReLU(),
            nn.Linear(128, 10),         # logits for 10 classses
        )

    def forward(self, x):
        return self.net(x)      # forward through layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partitions', required=True)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    # load lists of (path, label)
    with open(os.path.join(args.partitions, 'train.json'),'r') as f:
        train_idx = json.load(f)
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)

    # build loaders
    train_ds = FlatMNIST(train_idx)
    val_ds   = FlatMNIST(val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)    # move model to device
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)   #optimizer
    loss_fn = nn.CrossEntropyLoss() #loss for classification

    # train LOOP
    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            logits = model(xb)  # forward pass
            loss = loss_fn(logits, yb)  # compute loss
            opt.zero_grad() # clear old grads
            loss.backward() # backprop
            opt.step()  # upgrade params

        # eval LOOP
        model.eval()
        correct = total = 0
        preds_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), torch.tensor(yb).to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)    # get predicted class ids
                preds_all.append(preds.cpu().numpy())
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct/total
        print(f"Epoch {ep+1}/{args.epochs} - val acc {acc:.4f}")

    # save artifacts
    os.makedirs('experiments', exist_ok=True)
    torch.save(model.state_dict(), 'experiments/mlp_torch.pt')
    np.save('experiments/mlp_torch_val_preds.npy', np.concatenate(preds_all))
    print("Saved model and predictions to experiments/.")

if __name__ == '__main__':
    main()
