# app.py
import json
import streamlit as st


import orjson, pandas as pd

# Load JSON quickly
with open("C:/Users/dines/Projects/adc_def/abc/conversations.json", "rb") as f:
    data = orjson.loads(f.read())

# If it's a list of dicts
if isinstance(data, list):
    df = pd.DataFrame(data)
    print(df.columns)
    print(df.head())

    # Filter what you want
    print(df.head(10))
