import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "09_meta_ensemble")
DATA_DIR = os.path.join(META_DIR, "data")
MODELS_DIR = os.path.join(META_DIR, "models")

class MetaMLP(nn.Module):
    def __init__(self, input_dim=3200, output_dim=800):
        super(MetaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def meta_loss(recon, target):
    mse = nn.functional.mse_loss(recon, target, reduction='mean')
    # Use derivative loss here too to keep results clean
    diff_recon = recon[:, 1:] - recon[:, :-1]
    diff_target = target[:, 1:] - target[:, :-1]
    deriv_mse = nn.functional.mse_loss(diff_recon, diff_target, reduction='mean')
    return mse + 10.0 * deriv_mse

def train_meta():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training Meta-MLP Ensemble on {DEVICE}")
    
    # 1. Load Data
    X = np.load(os.path.join(DATA_DIR, "ensemble_inputs.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "ensemble_targets.npy")).astype(np.float32)
    
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 2. Scaling (Important for the 3200 inputs)
    scaler_x = StandardScaler()
    scaler_x.fit(X[train_idx])
    X = scaler_x.transform(X)
    
    scaler_y = StandardScaler()
    scaler_y.fit(y[train_idx])
    y = scaler_y.transform(y)
    
    # 3. Dataloaders
    train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 4. Model
    model = MetaMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for stability
    
    # 5. Training
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_val_loss = float('inf')
    epochs = 100
    patience = 10
    no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = meta_loss(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = meta_loss(out, by)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/100] Meta Val Loss: {avg_val_loss:.6f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "meta_ensemble_best.pth"))
            np.save(os.path.join(MODELS_DIR, "scaler_meta_x_mean.npy"), scaler_x.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_meta_x_scale.npy"), scaler_x.scale_)
            np.save(os.path.join(MODELS_DIR, "scaler_meta_y_mean.npy"), scaler_y.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_meta_y_scale.npy"), scaler_y.scale_)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
                
    print("Meta-MLP training complete.")

if __name__ == "__main__":
    train_meta()
