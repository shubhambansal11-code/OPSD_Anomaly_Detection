from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
#from src/events import point_flags_to_events

ROOT=Path(__file__).resolve().parent.parent
OUTPUTS_DIR=ROOT/"outputs"
PLOT_DIR=OUTPUTS_DIR/"diagnostics"

def load_series(path:Path, column_name:str):
    df=pd.read_csv(path, index_col=0, parse_dates=True)
    return df[column_name]

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # residual=pd.read_csv(OUTPUTS_DIR/"residual_series.csv", index_col=0, parse_dates=True)
    # ae_scores=pd.read_csv(OUTPUTS_DIR/"ae_scores.csv", index_col=0, parse_dates=True)
    # ae_flags=pd.read_csv(OUTPUTS_DIR/"ae_flags.csv", index_col=0, parse_dates=True)
    # z_flags=pd.read_csv(OUTPUTS_DIR/"z_flags.csv", index_col=0, parse_dates=True)
    residual=load_series(OUTPUTS_DIR/"residual_series.csv", "residual")
    ae_scores=load_series(OUTPUTS_DIR/"ae_scores.csv", "ae_recon_mse")
    ae_flags=load_series(OUTPUTS_DIR/"ae_flags.csv", "is_anomaly_ae").astype(int)
    z_flags=load_series(OUTPUTS_DIR/"z_flags.csv", "is_anomaly_zscore").astype(int)
    if_flags=load_series(OUTPUTS_DIR/"if_flags.csv", "is_anomaly_iforest").astype(int)
    ae_events=pd.read_csv(OUTPUTS_DIR/"ae_events.csv", parse_dates=["start_time", "end_time"])
    if_events=pd.read_csv(OUTPUTS_DIR/"if_events.csv", parse_dates=["start_time", "end_time"])
    z_events=pd.read_csv(OUTPUTS_DIR/"z_events.csv", parse_dates=["start_time", "end_time"])
    #overlap_df=pd.DataFrame({"ae_flag":ae_flags,"z_flag":z_flags}).fillna(0).astype(int)
    overlap_df=pd.concat([ae_flags.rename("ae_flag"),z_flags.rename("z_flag")], axis=1).fillna(0).astype(int)
    print("\nAE versus Z score overlap:")
    print(pd.crosstab(overlap_df["ae_flag"],overlap_df["z_flag"]))
    print("\nAE score summary:")
    print(ae_scores.describe())
    plt.figure(figsize=(10,4))
    ae_scores.hist(bins=50)
    plt.title("AE reconstruction error distribution")
    plt.savefig(PLOT_DIR/"ae_rec_error_distribution.png")
    plt.close()

    high_ae_points=ae_scores.loc[ae_flags==1].sort_values(ascending=False).head(7)
    for i, ts in enumerate(high_ae_points.index,start=1):
        #96 hour window centered around the anomaly
        start=ts-pd.Timedelta(hours=48)
        end=ts+pd.Timedelta(hours=48)
        window=residual[start:end]
        plt.figure(figsize=(10,4))
        plt.plot(window.index, window.values)
        #plt.axvline(ts, linestyle="")
        plt.axvline(ts, linestyle="--", color='red')
        plt.scatter([ts], [residual.loc[ts]])
        plt.title(f"Residual window {i}")
        plt.savefig(PLOT_DIR/f"ae_window_{i}.png")
        plt.close()
        print(f"\nHighest AE anomaly {i}")
        print(f"Timestamp:{ts}")
        print(f"AE score:{ae_scores.loc[ts]:.5f}")
        print(f"Residual:{residual.loc[ts]:.3f}")
    
    monthly_vol=residual.resample("M").std().rename("residual_std")
    monthly_vol.index=monthly_vol.index.to_period("M").to_timestamp()
    monthly_vol.to_csv(PLOT_DIR/"monthly_residual_volatility.csv",index=True)
    print("\nTop 10 months by residual volatility:")
    print(monthly_vol.sort_values(ascending=False).head(20))

    rolling_std=residual.rolling(24*7).std()
    plt.figure(figsize=(10,4))
    plt.plot(rolling_std.index,rolling_std.values)
    plt.title("Rolling residual volatility (7 day SD)")
    plt.savefig(PLOT_DIR/"rolling_residual_volatility.png")
    plt.close()

    summ=pd.DataFrame({"model":["Autoencoder","Isolation Forest","Z-score"],"anomaly_points":[int(ae_flags.sum()),int(if_flags.sum()),int(z_flags.sum())],
    "event_count":[len(ae_events),len(if_events),len(z_events)],}) 
    summ["points_per_event"]=summ["anomaly_points"]/summ["event_count"]
    summ.to_csv(PLOT_DIR/"points_vs_events_summary.csv",index=False)
    print("\nPoint counts vs event counts:")
    print(summ)

if __name__=="__main__":
    main()