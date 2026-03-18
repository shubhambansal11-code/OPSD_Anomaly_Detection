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
if model_choice == "Autoencoder":
    events=ae_events
else:
    events=z_events

# Metric Section
st.subheader("Overview")
col1, col2, col3 = st.columns(3)
model_key = "autoencoder" if model_choice=="Autoencoder" else "zscore"
selected_row = summary[summary["model"]==model_key]

col1.metric("Anomaly Points", int(selected_row["anomaly_points"].values[0]))
col2.metric("Anomaly Rate (%)", round(selected_row["anomaly_rate"].values[0] * 100, 2))
col3.metric("Total Events", int(selected_row["event_count"].values[0]))

st.markdown("---")

#Residual Timeline Plot
st.subheader("Residual Time Series with Anomalies")
fig = go.Figure()
#Residual line
fig.add_trace(go.Scatter(
    x=residual.index,
    y=residual["residual"],
    mode="lines",
    name="Residual"))

#Anomaly points
anom_times=events["start_time"]

#Safe alignment
anom_values = residual.reindex(anom_times)["residual"]

fig.add_trace(go.Scatter(x=anom_times,y=anom_values,mode="markers",name="Anomalies",marker=dict(size=6)))
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# Model Score Plot
st.subheader("Model Score")
if model_choice == "Autoencoder":
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=ae_scores.index,y=ae_scores["ae_recon_mse"],mode="lines",name="Reconstruction Error"))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Z-score visualization can be added here....")

st.markdown("---")

# Monthly Event Counts
st.subheader("Monthly Anomaly Events")
events["month"]=events["start_time"].dt.to_period("M").dt.to_timestamp()
events_per_month=events.groupby("month").size()
fig3=go.Figure()
fig3.add_trace(go.Bar(x=events_per_month.index, y=events_per_month.values,name="Events per Month"))
st.plotly_chart(fig3, use_container_width=True)