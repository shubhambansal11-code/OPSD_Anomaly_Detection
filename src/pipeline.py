# src/pipeline.py
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OPSD_PATH
from src.data_clean import build_df_de 
from src.residuals import compute_load_residual

from src.features import build_feature_matrix
from src.models.baselines import rolling_zscore, zscore_flags
from src.models.autoencoder import fit_autoencoder, score_autoencoder, flag_from_contamination
from src.events import point_flags_to_events


def run_pipeline(contamination=0.01, z_thresh=3.0, gap_hours=2):
    
    #load raw OPSD
    df=pd.read_csv(OPSD_PATH, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    df=df.sort_index()

    #keep only Germany core cols and fill
    df_de=build_df_de(df)

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

    return {"df_de": df_de, "residual": residual, "X": X, "z": z, "z_flags": z_flags, "z_events": z_events, "ae_scores": ae_scores, "ae_flags": ae_flags, "ae_events": ae_events, "ae_thr": ae_thr}