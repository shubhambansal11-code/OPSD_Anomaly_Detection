# OPSD Anomaly Detection

This repository contains the files and software to detect anomalous events (forecast failures) in Deutschland power grid data.

# Workflow

### Step 1: Load data

src/download_opsd.py downloads hourly power system data.


### Step 2: Data cleaning

src/data_clean.py selects relevant variables and fills missing values using time-based interpolation.


### Step 3: Residual engineering

src/residuals.py creates the core signal:

Residual = Actual Load − Forecast Load

This represents forecast error and is the main anomaly signal.


### Step 4: Feature engineering

src/features.py creates lag features and rolling statistics on the residual series.


### Step 5: Statistical baseline

models/baselines.py implements rolling z-score (3 sigma rule) to detect extreme deviations.


### Step 6: Deep learning model

models/autoencoder.py implements a PyTorch Autoencoder to detect anomalies via reconstruction error.


### Step 7: Event detection

src/events.py converts point anomalies into event clusters with start and end time  duration, and severity  


### Step 8: Pipeline orchestration

src/pipeline.py integrates all modules into a single workflow.

# How to Run

Step 1: Set up the dependencies listed in Requirements.txt. Install them with 

```
pip install -r Requirements.txt
```
Step 2: 

Run the pipeline (static analysis)

```
python src/pipeline.py
```

