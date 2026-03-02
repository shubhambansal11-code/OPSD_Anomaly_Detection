# src/events.py
import numpy as np
import pandas as pd

def point_flags_to_events(residual, flags, scores=None, gap_tolerance_hours=2):
   
    #align
    df=pd.DataFrame(index=flags.index)
    df["residual"]=residual.reindex(flags.index)
    df["flag"]=flags

    if scores is not None:
        df["score"]=scores.reindex(flags.index)

    anom_times=df.index[df["flag"]==1]
    if len(anom_times)==0:
        return pd.DataFrame()

    diffs=anom_times.to_series().diff().dt.total_seconds().div(3600)
    new_group=diffs.isna() | (diffs > gap_tolerance_hours)
    event_id=new_group.cumsum().values

    anom_only=df.loc[anom_times].copy()
    anom_only["event_id"]=event_id

    agg_dict={"start_time": ("residual", lambda s: s.index.min()),
        "end_time": ("residual", lambda s: s.index.max()),
        "duration_hours": ("residual", "count"),
        "max_abs_residual_MW": ("residual", lambda s: float(np.max(np.abs(s.values)))),
        "mean_abs_residual_MW": ("residual", lambda s: float(np.mean(np.abs(s.values)))),}

    if "score" in anom_only.columns:
        agg_dict["max_score"]=("score", "max")

    events=(anom_only.groupby("event_id").agg(**agg_dict).sort_values("start_time").reset_index(drop=True))
    return events