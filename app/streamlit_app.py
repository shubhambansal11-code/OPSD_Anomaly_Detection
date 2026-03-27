# streamlit_app.py

# Preliminary structure for testing and visualization of results.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

#Load data
summary=pd.read_csv("outputs/summary_metrics.csv")
residual=pd.read_csv("outputs/residual_series.csv",
    parse_dates=["utc_timestamp"],
    index_col="utc_timestamp")
ae_events = pd.read_csv("outputs/ae_events.csv", parse_dates=["start_time", "end_time"])
z_events=pd.read_csv("outputs/z_events.csv", parse_dates=["start_time", "end_time"])
ae_scores=pd.read_csv("outputs/ae_scores.csv", index_col=0, parse_dates=True)

#Sidebar User Inputs.. add few more inputs
st.sidebar.header("Inputs")
model_choice = st.sidebar.selectbox("Select Model",["Autoencoder", "Z-score"])

#Need to cross-check this further
date_min=residual.index.min().date()
date_max=residual.index.max().date()
date_range=st.sidebar.date_input("Select Date Range", value=(date_min, date_max),min_value=date_min,max_value=date_max)

if isinstance(date_range, tuple) and len(date_range)==2:
    start_date=pd.to_datetime(date_range[0])
    end_date=pd.to_datetime(date_range[1])
else:
    start_date=pd.to_datetime(date_min)
    end_date=pd.to_datetime(date_max)

if residual.index.tz is not None:
    start_date=start_date.tz_localize(residual.index.tz)
    end_date=end_date.tz_localize(residual.index.tz)

#Title and Description
st.title("OPSD Anomaly Detection Dashboard")
#Add also about iso forests description
st.write("""
This dashboard analyzes forecast failure events in the German power system 
using data from Open Power System Data (OPSD).

Several approaches were explored:
- Statistical baseline (Z-score)
- Deep learning model (Autoencoder)

Anomalies represent significant deviations between actual and forecasted load.
""")

#Select data based on model
if model_choice=="Autoencoder":
    events=ae_events
else:
    events=z_events

residual_filtered=residual.loc[start_date:end_date].copy()
events=events[(events["start_time"] >= start_date)&(events["start_time"]<=end_date)].copy()

if model_choice=="Autoencoder":
    ae_scores_filtered=ae_scores.loc[start_date:end_date].copy()

#Severity Threshold Slider
#max_sev=float(events["max_abs_residual_MW"].max()) if not events.empty else 1000.0
#severity_threshold=st.sidebar.slider("Minimum Event Severity (MW)",min_value=0.0,max_value=max_sev,value=0.0)

# Metric Section
st.subheader("Overview")
col1, col2, col3 = st.columns(3)
model_key = "autoencoder" if model_choice=="Autoencoder" else "zscore"
selected_row = summary[summary["model"]==model_key]

#safety check
if selected_row.empty:
    st.error("No summary row found for model: {model_key}")
    st.stop()

col1.metric("Anomaly Points", int(selected_row["anomaly_points"].values[0]))
col2.metric("Anomaly Rate (%)", round(selected_row["anomaly_rate"].values[0] * 100, 2))
col3.metric("Total Events", int(selected_row["event_count"].values[0]))

st.markdown("---")

#Residual Timeline Plot
st.subheader("Residual Time Series with Anomalies")
fig=go.Figure()
#Residual line
fig.add_trace(go.Scatter(
    x=residual_filtered.index,
    y=residual_filtered["residual"],
    mode="lines",
    name="Residual"))

#Anomaly points
anom_times=events["start_time"]

#Safe alignment
common_index=residual_filtered.index.intersection(anom_times)
anom_values=residual_filtered.loc[common_index, "residual"]
#anom_values = residual.reindex(anom_times)["residual"]

fig.add_trace(go.Scatter(x=common_index,y=anom_values,mode="markers",name="Anomalies",marker=dict(size=6)))
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# Model Score Plot
st.subheader("Model Score")
if model_choice == "Autoencoder":
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=ae_scores.index,y=ae_scores["ae_recon_mse"],mode="lines",name="Reconstruction Error"))
    st.plotly_chart(fig2, use_container_width=True)
else:
    z_series_full=residual["residual"].copy()
    z_mean=z_series_full.rolling(24*7).mean()
    z_std=z_series_full.rolling(24*7).std()
    z_full=(z_series_full-z_mean)/z_std
    z_filtered=z_full.loc[start_date:end_date]
    #z_plot=(z_series_full-z_mean)/z_std
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=z_filtered.index,y=z_filtered,mode="lines",name="z-score"))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Monthly Event Counts
st.subheader("Monthly Anomaly Events")
events["month"]=events["start_time"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
events_per_month=events.groupby("month").size()
fig3=go.Figure()
fig3.add_trace(go.Bar(x=events_per_month.index, y=events_per_month.values,name="Events per Month"))
st.plotly_chart(fig3, use_container_width=True)