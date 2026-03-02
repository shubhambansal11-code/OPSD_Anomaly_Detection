# src/models/baselines.py
import numpy as np
import pandas as pd

def rolling_zscore(residual, window=24*7):
    residual=residual.astype(float)
    mean=residual.rolling(window).mean()
    std=residual.rolling(window).std()
    z=(residual-mean)/std
    return z.rename("residual_z")

def zscore_flags(z, threshold=3.0):
    return (np.abs(z) > threshold).astype(int).rename("is_anomaly_zscore")