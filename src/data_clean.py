import pandas as pd

#Core columns from OPSD
CORE_DE_COLS=[
    "DE_load_actual_entsoe_transparency",
    "DE_load_forecast_entsoe_transparency",
    "DE_solar_generation_actual",
    "DE_wind_generation_actual",]

#Return all columns containing DE
def get_de_columns(df):
    return [c for c in df.columns if "DE_" in c]

#Select core colunns
def select_core_de(df):
    return df[CORE_DE_COLS].copy()

#Ensure date-time index and sort
def ensure_datetime_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index=pd.to_datetime(df.index)
    return df.sort_index()

#Interpolate and fill
def fill_time_series_gaps(df):
    df=ensure_datetime_index(df)
    df=df.interpolate(method="time", limit_direction="both")
    df=df.ffill().bfill()
    return df