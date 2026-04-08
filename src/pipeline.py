# src/pipeline.py
import os

import pandas as pd
import matplotlib.pyplot as plt

from config import OPSD_PATH
from data_clean import select_core_de, ensure_datetime_index
from residuals import compute_load_residual, fill_residual_gaps

from features import build_feature_matrix
from models.baselines import rolling_zscore, zscore_flags
from models.autoencoder import fit_autoencoder, score_autoencoder, flag_from_contamination, save_autoencoder
#from models.autoencoder import fit_autoencoder, score_autoencoder, save_autoencoder, threshold_from_contamination, flag_from_threshold
from models.isoforests import (fit_isolation_forest, score_isolation_forest, predict_isolation_forest, flag_isolation_forest, save_isolation_forest,)
from events import point_flags_to_events
from metrics import (events_per_month, duration_stats, severity_per_month, anomaly_rate, top_severe_events)

def run_pipeline(contamination=0.01, z_thresh=3.0, gap_hours=2, train_frac=0.8):
    
    #load raw OPSD
    df=pd.read_csv(OPSD_PATH, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    df=df.sort_index()

    #keep only Germany core cols and fill
    df_de = select_core_de(df)
    df_de = ensure_datetime_index(df_de)

    #residual
    residual=compute_load_residual(df_de)
    residual=fill_residual_gaps(residual)

    #features X
    X=build_feature_matrix(residual)

    # split_idx=int(len(X) * train_frac)
    # X_train=X.iloc[:split_idx].copy()
    # X_test=X.iloc[split_idx:].copy()
    # split_time=X.index[split_idx]

    #z-baseline
    z=rolling_zscore(residual.reindex(X.index))
    z_flags=zscore_flags(z, threshold=z_thresh)
    z_events=point_flags_to_events(residual, z_flags, scores=z, gap_tolerance_hours=gap_hours)

    # Isolation Forest
    if_model=fit_isolation_forest(X, contamination=contamination, n_estimators=400, random_state=42)
    if_scores=score_isolation_forest(if_model, X)
    if_pred=predict_isolation_forest(if_model, X)
    if_flags=flag_isolation_forest(if_pred)
    if_events=point_flags_to_events(residual, if_flags, scores=if_scores, gap_tolerance_hours=gap_hours)

    #fit on train only
    # if_model=fit_isolation_forest(X_train, contamination=contamination, n_estimators=400, random_state=42,)

    # score train,test,full separately
    # if_train_scores=score_isolation_forest(if_model, X_train)
    # if_test_scores=score_isolation_forest(if_model, X_test)
    # if_scores=score_isolation_forest(if_model, X)   

    # if_train_pred=predict_isolation_forest(if_model, X_train)
    # if_test_pred=predict_isolation_forest(if_model, X_test)
    # if_pred=predict_isolation_forest(if_model, X)  

    # if_train_flags=flag_isolation_forest(if_train_pred)
    # if_test_flags=flag_isolation_forest(if_test_pred)
    # if_flags=flag_isolation_forest(if_pred)

    #if_events=point_flags_to_events(residual, if_flags, scores=if_scores, gap_tolerance_hours=gap_hours)

    #Autoencoder
    model, stats=fit_autoencoder(X, epochs=8, batch_size=200, lr=1e-3, random_state=42)
    ae_scores=score_autoencoder(model, stats, X)
    ae_flags, ae_thr=flag_from_contamination(ae_scores, contamination=contamination)
    ae_events=point_flags_to_events(residual, ae_flags, scores=ae_scores, gap_tolerance_hours=gap_hours)
    
    #fit on train only
    # model, stats=fit_autoencoder(X_train,epochs=8,batch_size=200,lr=1e-3,random_state=42,)

    #score train,test,full separately
    # ae_train_scores=score_autoencoder(model, stats, X_train)
    # ae_test_scores=score_autoencoder(model, stats, X_test)
    # ae_scores=score_autoencoder(model, stats, X)   # full history for visualization

    # threshold from train only
    # ae_thr=threshold_from_contamination(ae_train_scores, contamination=contamination)

    # apply same threshold to train/test/full
    # ae_train_flags=flag_from_threshold(ae_train_scores, ae_thr)
    # ae_test_flags=flag_from_threshold(ae_test_scores, ae_thr)
    # ae_flags=flag_from_threshold(ae_scores, ae_thr)

    # ae_events=point_flags_to_events(residual, ae_flags, scores=ae_scores, gap_tolerance_hours=gap_hours)

    os.makedirs("outputs", exist_ok=True) 
    #save model+stats
    save_autoencoder(model, stats, "outputs/autoencoder.pt", "outputs/autoencoder_stats.pkl")
    #save_isolation_forest(if_model, "outputs/isolation_forest.pkl")

    # Metrics
    z_event_counts=events_per_month(z_events)
    z_duration=duration_stats(z_events)
    z_severity=severity_per_month(z_events)
    z_rate=anomaly_rate(z_flags)

    if_event_counts=events_per_month(if_events)
    if_duration=duration_stats(if_events)
    if_severity=severity_per_month(if_events)
    if_rate=anomaly_rate(if_flags)

    ae_event_counts=events_per_month(ae_events)
    ae_duration=duration_stats(ae_events)
    ae_severity=severity_per_month(ae_events)
    ae_rate=anomaly_rate(ae_flags)

    ae_scores.to_csv("outputs/ae_scores.csv", index=True)
    ae_flags.to_csv("outputs/ae_flags.csv", index=True)   
    if_scores.to_csv("outputs/if_scores.csv", index=True)
    if_flags.to_csv("outputs/if_flags.csv", index=True)       
    z.to_csv("outputs/z_scores.csv", index=True)                 
    z_flags.to_csv("outputs/z_flags.csv", index=True)            
    ae_events.to_csv("outputs/ae_events.csv", index=False)
    z_events.to_csv("outputs/z_events.csv", index=False)
    if_events.to_csv("outputs/if_events.csv", index=False)

    residual.reindex(X.index).to_frame(name="residual").to_csv("outputs/residual_series.csv", index=True)
    #residual.to_frame(name="residual").to_csv("outputs/residual_series.csv", index=True)
    
    # save train/test comparison outputs
    # ae_train_scores.to_csv("outputs/ae_train_scores.csv", index=True)
    # ae_test_scores.to_csv("outputs/ae_test_scores.csv", index=True)
    # ae_train_flags.to_csv("outputs/ae_train_flags.csv", index=True)
    # ae_test_flags.to_csv("outputs/ae_test_flags.csv", index=True)

    # if_train_scores.to_csv("outputs/if_train_scores.csv", index=True)
    # if_test_scores.to_csv("outputs/if_test_scores.csv", index=True)
    # if_train_flags.to_csv("outputs/if_train_flags.csv", index=True)
    # if_test_flags.to_csv("outputs/if_test_flags.csv", index=True)

    #train/test event summaries
    # ae_train_events = point_flags_to_events(residual, ae_train_flags, scores=ae_train_scores, gap_tolerance_hours=gap_hours)
    # ae_test_events = point_flags_to_events(residual, ae_test_flags, scores=ae_test_scores, gap_tolerance_hours=gap_hours)

    # if_train_events = point_flags_to_events(residual, if_train_flags, scores=if_train_scores, gap_tolerance_hours=gap_hours)
    # if_test_events = point_flags_to_events(residual, if_test_flags, scores=if_test_scores, gap_tolerance_hours=gap_hours)

    # ae_train_events.to_csv("outputs/ae_train_events.csv", index=False)
    # ae_test_events.to_csv("outputs/ae_test_events.csv", index=False)
    # if_train_events.to_csv("outputs/if_train_events.csv", index=False)
    # if_test_events.to_csv("outputs/if_test_events.csv", index=False)

    pd.DataFrame({"model": ["autoencoder"], "threshold": [ae_thr],"contamination": [contamination],}).to_csv("outputs/ae_threshold.csv", index=False)
    #pd.DataFrame({"model": ["autoencoder"], "threshold": [ae_thr], "contamination": [contamination], "train_fraction": [train_frac],"split_time": [split_time],}).to_csv("outputs/ae_threshold.csv", index=False)
    pd.DataFrame({"contamination": [contamination],"z_thresh": [z_thresh],"gap_hours": [gap_hours],}).to_csv("outputs/run_config.csv", index=False)
    #pd.DataFrame({"contamination": [contamination],"z_thresh": [z_thresh], "gap_hours": [gap_hours], "train_fraction": [train_frac], "split_time": [split_time],}).to_csv("outputs/run_config.csv", index=False)

    summary_metrics = pd.DataFrame({"model":["zscore", "iforest", "autoencoder"], "anomaly_points":[int(z_flags.sum()), int(if_flags.sum()), int(ae_flags.sum())],
        "anomaly_rate":[float(z_rate), float(if_rate), float(ae_rate)], "event_count":[len(z_events), len(if_events), len(ae_events)],
        "avg_event_duration_hours":[float(z_events["duration_hours"].mean()) if not z_events.empty else 0.0,
            float(if_events["duration_hours"].mean()) if not if_events.empty else 0.0,
            float(ae_events["duration_hours"].mean()) if not ae_events.empty else 0.0,],
        "avg_event_severity_mw": [float(z_events["max_abs_residual_MW"].mean()) if not z_events.empty else 0.0,
            float(if_events["max_abs_residual_MW"].mean()) if not if_events.empty else 0.0,
            float(ae_events["max_abs_residual_MW"].mean()) if not ae_events.empty else 0.0,],})
    summary_metrics.to_csv("outputs/summary_metrics.csv", index=False)

    # split_summary_metrics = pd.DataFrame({ "model": ["autoencoder", "autoencoder", "iforest", "iforest"],
    #     "split": ["train", "test", "train", "test"], "anomaly_points": [int(ae_train_flags.sum()), int(ae_test_flags.sum()),
    #         int(if_train_flags.sum()), int(if_test_flags.sum()),],
    #     "anomaly_rate": [float(anomaly_rate(ae_train_flags)), float(anomaly_rate(ae_test_flags)), float(anomaly_rate(if_train_flags)),
    #         float(anomaly_rate(if_test_flags)),],
    #     "event_count": [len(ae_train_events), len(ae_test_events), len(if_train_events), len(if_test_events),],})
    # split_summary_metrics.to_csv("outputs/split_summary_metrics.csv", index=False)
    
    #Too many at the moment, try slimming them down a bit..
    return {"df_de": df_de, "residual": residual, "X": X, "z": z, "z_flags": z_flags, "z_events": z_events, "z_event_counts": z_event_counts,
        "z_duration": z_duration, "z_severity": z_severity, "if_scores": if_scores, "if_flags": if_flags, "if_events": if_events, "if_event_counts": if_event_counts,
        "if_duration": if_duration, "if_severity": if_severity, "ae_scores": ae_scores, "ae_flags": ae_flags, "ae_events": ae_events, "ae_event_counts": ae_event_counts, 
        "ae_duration": ae_duration, "ae_severity": ae_severity, "ae_thr": ae_thr}
    
    # return {"df_de": df_de, "residual": residual, "X": X, "X_train": X_train, "X_test": X_test, "split_time": split_time, "z": z,
    #     "z_flags": z_flags, "z_events": z_events, "z_event_counts": z_event_counts, "z_duration": z_duration,  "z_severity": z_severity,
    #     "if_scores": if_scores, "if_flags": if_flags, "if_events": if_events, "if_event_counts": if_event_counts, "if_duration": if_duration, "if_severity": if_severity,
    #     "if_train_scores": if_train_scores, "if_test_scores": if_test_scores, "if_train_flags": if_train_flags, "if_test_flags": if_test_flags,
    #     "ae_scores": ae_scores, "ae_flags": ae_flags, "ae_events": ae_events, "ae_event_counts": ae_event_counts, "ae_duration": ae_duration,
    #     "ae_severity": ae_severity, "ae_thr": ae_thr, "ae_train_scores": ae_train_scores, "ae_test_scores": ae_test_scores, "ae_train_flags": ae_train_flags,
    #     "ae_test_flags": ae_test_flags,}

if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline run complete")