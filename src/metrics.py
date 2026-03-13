import numpy as np
import pandas as pd

def events_per_month(event_summary):
    #Count anomalous events per month on the basis of event summary
    if event_summary.empty:
        return pd.Series(dtype=float)
    event_summary=event_summary.copy()
    event_summary["month"]=event_summary["start_time"].dt.to_period("M").dt.to_timestamp()
    return event_summary.groupby("month").size().rename("events_per_month")

def duration_stats(event_summary):
    #Describe event durations in hours
    if event_summary.empty:
        return pd.Series(dtype=float)
    return event_summary["duration_hours"].describe()

def severity_per_month(event_summary):
    #Average max absolute residual per month
    if event_summary.empty:
        return pd.Series(dtype=float)
    event_summary=event_summary.copy()
    event_summary["month"]=event_summary["start_time"].dt.to_period("M").dt.to_timestamp()
    return (event_summary.groupby("month")["max_abs_residual_MW"].mean().rename("avg_max_abs_residual_MW"))

def anomaly_rate(flags):
    #Proportion of time points flagged as anomalous
    if len(flags)==0:
        return np.nan
    return float(flags.mean())

def top_severe_events(event_summary, n=10):
    #Top n severe events by max absolute residual
    if event_summary.empty:
        return event_summary
    return event_summary.sort_values("max_abs_residual_MW", ascending=False).head(n)

def early_detection_hours(model_anom_times, baseline_events, lookback_hours=24):
    #Mean lead time in hours:
    #to check how much earlier anomalies appear before baseline event start
    if len(model_anom_times)==0 or baseline_events.empty:
        return np.nan
    lookback=pd.Timedelta(hours=lookback_hours)
    lead_times=[]
    for _, row in baseline_events.iterrows():
        start=row["start_time"]
        end=row["end_time"]
        model_window=model_anom_times[(model_anom_times >= start-lookback) & (model_anom_times <= end)]
        if len(model_window) > 0:
            lead_hours=(start-model_window.min()).total_seconds()/3600.0
            lead_hours=max(0.0, lead_hours)
            lead_times.append(lead_hours)
    if len(lead_times) == 0:
        return np.nan
    return float(np.mean(lead_times))