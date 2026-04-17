# OPSD Anomaly Detection

This repository contains the code and tools to detect anomalous events (i.e., deviations between actual and forecasted load) in the German power system using data from Open Power System Data (OPSD).

# Workflow

### Step 1: Load data

src/download_opsd.py downloads hourly power system data.


### Step 2: Data cleaning

src/data_clean.py selects relevant variables.


### Step 3: Residual engineering

src/residuals.py creates the core signal:

Residual = Actual Load − Forecast Load

This represents forecast error and is the main anomaly signal. The missing values are filled using time-based interpolation.

### Step 4: Feature engineering

src/features.py creates lag features and rolling statistics on the residual series.


### Step 5: Statistical baseline

models/baselines.py implements rolling z-score (3 sigma rule) to detect extreme deviations.

### Step 6: Machine learning model

models/isoforests.py implements an Isolation Forest model to detect anomalies based on how isolated a data point is from the learned normal data distribution.


### Step 7: Deep learning model

models/autoencoder.py implements a PyTorch Autoencoder to detect anomalies via reconstruction error.


### Step 8: Event detection

src/events.py converts point anomalies into event clusters with start and end time duration, and severity  


### Step 9: Pipeline orchestration

src/pipeline.py integrates all modules into a single workflow.

# How to Run

Step 1: Set up the dependencies listed in Requirements.txt. Install them with 

```
pip install -r requirements.txt
```
Step 2: 

Run the pipeline (static analysis)

```
python src/pipeline.py
```

Step 3: 

Produce the dashboard

```
python -m streamlit run app/streamlit_app.py
```