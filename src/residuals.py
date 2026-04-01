import pandas as pd
import numpy as np

#Residuals
def compute_load_residual(df):
    return (df["DE_load_actual_entsoe_transparency"]-df["DE_load_forecast_entsoe_transparency"])

#Fill gaps in residuals
def fill_residual_gaps(residual):
    residual=residual.interpolate(method="time").ffill().bfill()
    return residual

#Rolling z-score for anomaly baseline
def rolling_zscore(series, window=24*7): 
    mean=series.rolling(window).mean()
    std=series.rolling(window).std()
    return (series-mean)/std

#Binary anomaly flag based on z-score
def flag_zscore_anomalies(zscore_series, threshold=3):
    return (np.abs(zscore_series) > threshold).astype(int)

#Full residual, z-score, anomaly flag 
def build_residual_dataframe(df, window=24*7, threshold=3):
    residual=compute_load_residual(df)
    z=rolling_zscore(residual, window=window)
    is_anom=flag_zscore_anomalies(z, threshold=threshold)
    return pd.DataFrame({"residual": residual,"zscore": z,"is_anomaly": is_anom})