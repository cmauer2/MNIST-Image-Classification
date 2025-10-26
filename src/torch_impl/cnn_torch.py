import os, argparse, json, torch, numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom dataset class for MNIST 
class MNIST2D(Dataset):
    def __init__(self, index_list, normalize_to_neg1=False):
        self.index_list = index_list    # store list of (path, label)
        if normalize_to_neg1:
            self.t = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
        else:
            self.t = transforms.ToTensor()      #default [0,1] tensor

    def __len__(self): return len(self.index_list)      # of samples

    def __getitem__(self, idx):
        path, lab = self.index_list[idx]    # grab path and label
        x = Image.open(path).convert('L')   # open as gray scale
        x = self.t(x)                       # to torch tensor (1, 28, 28)
        return x, lab                       #return image tensor and label

# small CNN for 28x28 MNIST
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample > 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample again > 7x7
        )
        self.classifier = nn.Sequential(        # map features to class scores
            nn.Flatten(),                       # (64, 7, 7) -> 3136
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),     # 10 digits
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    ap = argparse.ArgumentParser()      # command line options
    ap.add_argument('--partitions', required=True)  # folder with train/val json
    ap.add_argument('--lr', type=float, default=0.01)   # learning rate
    ap.add_argument('--epochs', type=int, default=10)   # training epochs
    ap.add_argument('--batch_size', type=int, default=128)  # batch size
    ap.add_argument('--normalize_to_neg1', action='store_true')
    args = ap.parse_args()

    # load (path, label) lists
    with open(os.path.join(args.partitions, 'train.json'),'r') as f:
        train_idx = json.load(f)
    with open(os.path.join(args.partitions, 'val.json'),'r') as f:
        val_idx = json.load(f)

    # build datasets a & loaders
    train_ds = MNIST2D(train_idx, normalize_to_neg1=args.normalize_to_neg1)
    val_ds   = MNIST2D(val_idx, normalize_to_neg1=args.normalize_to_neg1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)   # shuffle
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)                   # no shuffle for eval

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # gpu if available
    model = SmallCNN().to(device)                                               # move model to device
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)         # optimizer
    loss_fn = nn.CrossEntropyLoss()                                             # classification loss

    # TRAINING LOOP
    for ep in range(args.epochs):
        model.train()       # enable training mode
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation loop (no gradient)
        model.eval()
        correct = total = 0
        preds_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), torch.tensor(yb).to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                preds_all.append(preds.cpu().numpy())
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct/total
        print(f"Epoch {ep+1}/{args.epochs} - val acc {acc:.4f}")

    # save model weights and validation predicts
    os.makedirs('experiments', exist_ok=True)
    torch.save(model.state_dict(), 'experiments/cnn_torch.pt')
    np.save('experiments/cnn_torch_val_preds.npy', np.concatenate(preds_all))
    print("Saved model and predictions to experiments/.")

if __name__ == '__main__':
    main()
