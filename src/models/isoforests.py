#src/models/isoforests.py
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def fit_isolation_forest(X,contamination=0.01,n_estimators=400,random_state=42):
    model=IsolationForest(n_estimators=n_estimators,contamination=contamination,random_state=random_state,)
    model.fit(X)
    return model

def score_isolation_forest(model,X):
    scores=pd.Series(model.decision_function(X),index=X.index,name="iforest_score",)
    return scores

def predict_isolation_forest(model,X):
    pred=pd.Series(model.predict(X),index=X.index,name="iforest_pred",)
    return pred

def flag_isolation_forest(pred):
    flags=(pred == -1).astype(int).rename("is_anomaly_iforest")
    return flags

def save_isolation_forest(model,model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def load_isolation_forest(model_path):
    with open(model_path, "rb") as f:
        model=pickle.load(f)
    return model