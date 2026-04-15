# streamlit_app.py

# Preliminary structure for testing and visualization of results.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

#Load data
summary=pd.read_csv("outputs/summary_metrics.csv")
residual=pd.read_csv("outputs/residual_series.csv",
    parse_dates=["utc_timestamp"],
    index_col="utc_timestamp")
ae_events = pd.read_csv("outputs/ae_events.csv", parse_dates=["start_time", "end_time"])
if_events = pd.read_csv("outputs/if_events.csv", parse_dates=["start_time", "end_time"])
z_events=pd.read_csv("outputs/z_events.csv", parse_dates=["start_time", "end_time"])
#ae_scores=pd.read_csv("outputs/ae_scores.csv", index_col=0, parse_dates=True)
#if_scores=pd.read_csv("outputs/if_scores.csv", index_col=0, parse_dates=True)
#z_scores=pd.read_csv("outputs/z_scores.csv", index_col=0, parse_dates=True)

st.sidebar.header("Inputs")
#model_choice = st.sidebar.selectbox("Select Model",["Autoencoder","Isolation Forest","Z-score"])
show_if=st.sidebar.checkbox("Compare with Isolation Forest", value=False)
show_z=st.sidebar.checkbox("Compare with Z-score", value=False) 

# date_min=residual.index.min().date()
# date_max=residual.index.max().date()
# date_range=st.sidebar.date_input("Select Date Range", value=(date_min, date_max),min_value=date_min,max_value=date_max)

st.sidebar.markdown("""
**Notes**

1. The residual is defined as: Residual = Actual Load - Forecasted Load

2. The methods define anomalous timestamps differently:

    a. Z-score flags timestamps where the residual is unusually far from its recent rolling average  
      (using a 1-week rolling window and threshold |z| > 3).

    b. Autoencoder operates on a feature vector that includes the current residual, the lagged residuals and the rolling statistics.  
      It flags timestamps where this feature pattern deviates from patterns learnt during training, resulting in high reconstruction error. 
      Approximately the top 1% highest reconstruction errors are marked as anomalies.

    c. Isolation Forest uses the same feature representation as the autoencoder. 
      However, it flags timestamps that lie in sparse or structurally different regions of this feature space, meaning they can be isolated with fewer splits.  
      Approximately the top 1% most isolated points are marked as anomalies.

3. Anomalous events are formed by grouping anomalous timestamps that occur within 2 hours of each other.

4. In the time-series plot, markers indicate the start time of each detected event.
""")

# if isinstance(date_range, tuple) and len(date_range)==2:
#     start_date=pd.to_datetime(date_range[0])
#     end_date=pd.to_datetime(date_range[1])
# else:
#     start_date=pd.to_datetime(date_min)
#     end_date=pd.to_datetime(date_max)

# if residual.index.tz is not None:
#     start_date=start_date.tz_localize(residual.index.tz)
#     end_date=end_date.tz_localize(residual.index.tz)

#Title and Description
st.title("OPSD Anomaly Detection")
st.write("""
This dashboard analyzes forecast failure events in the German power system 
using data from Open Power System Data (OPSD).         

The main view shows the deep leaning autoencoder model to detect anomalous events. 
Comparison with Isolation Forest and Z-score methods can be enabled from the sidebar.

Anomalies represent significant deviations between actual and forecasted load. 
""")

# ae_events_filtered=ae_events[(ae_events["start_time"]>=start_date) & (ae_events["start_time"]<=end_date)].copy()
# if_events_filtered=if_events[(if_events["start_time"]>=start_date) & (if_events["start_time"]<=end_date)].copy()
# z_events_filtered=z_events[(z_events["start_time"]>=start_date) & (z_events["start_time"]<=end_date)].copy()
# residual_filtered=residual[(residual.index>=start_date) & (residual.index<=end_date)].copy()

#use full saved data directly
residual_plot=residual.copy()
ae_events_plot=ae_events.copy()
if_events_plot=if_events.copy()
z_events_plot=z_events.copy()

# st.subheader("Overview")
# ae_summary_row = summary[summary["model"]=="autoencoder"]
# if ae_summary_row.empty:
#     st.error("No summary row found for autoencoder")
#     st.stop()
# st.metric("Autoencoder Total Events", int(ae_summary_row["event_count"].values[0]))
# st.markdown("---")

#Residual Timeseries Plot
st.subheader("Residual Time Series with Anomalous Events")
fig = go.Figure()

#Residual line
fig.add_trace(go.Scatter(x=residual_plot.index,y=residual_plot["residual"],mode="lines",name="Residual",line=dict(width=1, color="blue"), opacity=0.8))

ae_times=ae_events_plot["start_time"]
ae_common_index = residual_plot.index.intersection(ae_times)
ae_values = residual_plot.loc[ae_common_index, "residual"]

fig.add_trace(go.Scatter(x=ae_common_index,y=ae_values,mode="markers",name="Autoencoder Events",marker=dict(size=5, color="black")))

if show_if:
    if_times=if_events_plot["start_time"]
    if_common_index=residual_plot.index.intersection(if_times)
    if_values=residual_plot.loc[if_common_index, "residual"]
    fig.add_trace(go.Scatter(x=if_common_index,y=if_values,mode="markers",name="Isolation Forest Events",marker=dict(size=5, color="orange", symbol="diamond")))

if show_z:
    z_times=z_events_plot["start_time"]
    z_common_index=residual_plot.index.intersection(z_times)
    z_values=residual_plot.loc[z_common_index, "residual"]
    fig.add_trace(go.Scatter(x=z_common_index,y=z_values,mode="markers",name="Z-score Events",marker=dict(size=5, color="red", opacity=0.8, symbol="x")))

fig.update_layout(height=600, xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"""**Note:** Detected anomalies may not overlap and should be viewed as complementary signals capturing different types of anamolous behaviour. Z-score identifies short-lived extreme deviations in magnitude, Isolation Forest detects rare observations in the feature space, and the Autoencoder captures sustained unusual temporal patterns.""")
st.markdown("---")

#Monthly Event Counts
st.subheader("Monthly Anomalous Events")
fig2=go.Figure()

if not ae_events_plot.empty:
    ae_events_plot=ae_events_plot.copy()
    ae_events_plot["month"]=ae_events_plot["start_time"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    ae_events_per_month=ae_events_plot.groupby("month").size()
    fig2.add_trace(go.Bar(x=ae_events_per_month.index,y=ae_events_per_month.values,marker_color="black",name="Autoencoder"))

if show_if and not if_events_plot.empty:
    if_events_plot=if_events_plot.copy()
    if_events_plot["month"]=if_events_plot["start_time"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    if_events_per_month=if_events_plot.groupby("month").size()
    fig2.add_trace(go.Bar(x=if_events_per_month.index,y=if_events_per_month.values,marker_color="orange",name="Isolation Forest"))

if show_z and not z_events_plot.empty:
    z_events_plot=z_events_plot.copy()
    z_events_plot["month"]=z_events_plot["start_time"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    z_events_per_month=z_events_plot.groupby("month").size()
    fig2.add_trace(go.Bar(x=z_events_per_month.index,y=z_events_per_month.values,marker_color="red",name="Z-score"))

fig2.update_layout(height=600)
st.plotly_chart(fig2, use_container_width=True)
st.markdown(f"""**Note:** Months with higher event counts often reflect periods with increased residual variability, as well as how closely anomaly points occur in time given the 2 hour grouping rule.""")
