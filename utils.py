import pandas as pd

def df_to_dicts(df):
    return df.to_dict(orient='list')