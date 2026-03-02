# src/features.py
import pandas as pd
def build_feature_matrix(residual):

    residual=residual.astype(float)
    X=pd.DataFrame(index=residual.index)
    X["resid"]=residual

    #lags
    for lag in [1, 6, 24]:
        X[f"resid_lag_{lag}"]=residual.shift(lag)

    #rolling stats
    X["resid_roll_mean_24"]=residual.rolling(24).mean()
    X["resid_roll_std_24"]=residual.rolling(24).std()

    X=X.dropna()
    return X