# src/models/autoencoder.py
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 8), nn.ReLU(),
            nn.Linear(8, 2), nn.ReLU(),
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, d),)
        
    def forward(self, x):
        return self.net(x)

def fit_autoencoder(X, epochs=8, batch_size=200, lr=1e-3, random_state=42):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    #standardize using full data stats
    mu=X.mean()
    sd=X.std().replace(0, 1.0)
    Xn=((X-mu)/sd).values.astype(np.float32)

    loader=DataLoader(TensorDataset(torch.from_numpy(Xn)), batch_size=batch_size, shuffle=True)

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=AE(d=X.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn=nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total=0.0
        for (xb,) in loader:
            xb=xb.to(device)
            recon=model(xb)
            loss=loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total+=loss.item()*len(xb)
        #print(f"epoch {epoch+1}: train_mse={total/len(Xn):.4f}")
    stats = {"mu": mu, "sd": sd, "device": device, "input_dim": X.shape[1],}
    return model, stats

def score_autoencoder(model, stats, X):
    mu, sd=stats["mu"], stats["sd"]
    device=stats["device"]
    Xn=((X-mu)/sd).values.astype(np.float32)
    model.eval()
    with torch.no_grad():
        xt=torch.from_numpy(Xn).to(device)
        recon=model(xt).cpu().numpy()
    err=np.mean((recon-Xn)**2, axis=1)
    return pd.Series(err, index=X.index, name="ae_recon_mse")

def flag_from_contamination(scores, contamination=0.01):
    thr=float(np.quantile(scores.values, 1-contamination))
    flags=(scores>thr).astype(int).rename("is_anomaly_ae")
    return flags, thr

def save_autoencoder(model, stats, model_path, stats_path):
    torch.save(model.state_dict(), model_path)
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

def load_autoencoder(model_path, stats_path, device=None):
    with open(stats_path, "rb") as f:
        stats=pickle.load(f)
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"

    model=AE(d=stats["input_dim"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    stats["device"]=device
    return model, stats