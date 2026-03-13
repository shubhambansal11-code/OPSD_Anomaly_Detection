# src/pipeline.py
import pandas as pd
import matplotlib.pyplot as plt

from config import OPSD_PATH
from data_clean import select_core_de, fill_time_series_gaps
from residuals import compute_load_residual

from features import build_feature_matrix
from models.baselines import rolling_zscore, zscore_flags
from models.autoencoder import fit_autoencoder, score_autoencoder, flag_from_contamination
from events import point_flags_to_events
from metrics import (events_per_month, duration_stats, severity_per_month, anomaly_rate, top_severe_events)

def run_pipeline(contamination=0.01, z_thresh=3.0, gap_hours=2):
    
    #load raw OPSD
    df=pd.read_csv(OPSD_PATH, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    df=df.sort_index()

    #keep only Germany core cols and fill
    df_de = select_core_de(df)
    df_de = fill_time_series_gaps(df_de)

    #residual
    residual=compute_load_residual(df_de)

    #features X
    X=build_feature_matrix(residual)

    #z-baseline
    z=rolling_zscore(residual.reindex(X.index))
    z_flags=zscore_flags(z, threshold=z_thresh)
    z_events=point_flags_to_events(residual, z_flags, scores=z, gap_tolerance_hours=gap_hours)

    #Autoencoder
    model, stats=fit_autoencoder(X, epochs=8, batch_size=200, lr=1e-3)
    ae_scores=score_autoencoder(model, stats, X)
    ae_flags, ae_thr=flag_from_contamination(ae_scores, contamination=contamination)
    ae_events=point_flags_to_events(residual, ae_flags, scores=ae_scores, gap_tolerance_hours=gap_hours)

    # Metrics
    z_event_counts=events_per_month(z_events)
    z_duration=duration_stats(z_events)
    z_severity=severity_per_month(z_events)
    z_rate=anomaly_rate(z_flags)

    ae_event_counts=events_per_month(ae_events)
    ae_duration=duration_stats(ae_events)
    ae_severity=severity_per_month(ae_events)
    ae_rate=anomaly_rate(ae_flags)
    
    #Too many at the moment, try slimming them down a bit..
    return {"df_de": df_de, "residual": residual, "X": X, "z": z, "z_flags": z_flags, "z_events": z_events, "z_event_counts": z_event_counts,
        "z_duration": z_duration, "z_severity": z_severity, "ae_scores": ae_scores, "ae_flags": ae_flags, "ae_events": ae_events,
        "ae_event_counts": ae_event_counts, "ae_duration": ae_duration, "ae_severity": ae_severity,
        "ae_thr": ae_thr}