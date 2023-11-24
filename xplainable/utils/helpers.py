import json
import pandas as pd
from ..utils.encoders import NpEncoder

def get_df_delta(df1, df2):
    """ Gets the delta between two dataframs"""
    changed_features = [c for c in df1.columns.intersection(
        df2.columns) if not df1[c].head(10).equals(df2[c].head(10))]

    rows_changed = {}
    for c in changed_features:
        if c in df1.columns and c in df2.columns:
            rows_changed[c] = (df1[c] != df2[c]).sum()
        
    output = {
        "drop": [col for col in df1.columns if col not in df2.columns],
        "add": [{"feature": col, "values": json.loads(
        df2[col].head(10).to_json(orient='records'))} for col in \
            df2.columns if col not in df1.columns],
        "update": [{
            "feature": col,
            "values": json.loads(df2[col].head(10).to_json(
                orient='records'))} for col in changed_features],
        "rows_affected": json.loads(json.dumps(rows_changed, cls=NpEncoder))
        }

    return output