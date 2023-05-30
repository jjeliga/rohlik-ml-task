import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import timedelta

from conf import (
    DATE_FORMAT,
    TRANSFORM_CONF,
    PROMOTION_CONF,
    ID_COL,
    DATE_COL,
    SALES_COL,
    MARGIN_COL,
    PRICE_COL,
    BUY_PRICE_COL
    )
from utils import (
    split_df,
    general_transform,
    init_transform,
    is_weekend,
    add_czech_holidays,
    identify_promotions
    )

# %% helper functions



def estimate_absolute_margin(df, margin_col, sales_col, window=14):
    """
    Estimates absolute margin received in a specified time `window`.
    Takes the last 2*window days of data, and averages the absolute margins.
    It is advised to use window values divisible by 7.
    """
    df_am1 = df[-2*window:-window]
    df_am2 = df[-window:]
    am1 = sum(df_am1[margin_col] * df_am1[sales_col])
    am2 = sum(df_am2[margin_col] * df_am2[sales_col])
    
    return (am1 + am2) / 2


def generate_future_data(df, date_col, date_format, h=7):
    """
    Returns dataframe for the next `h` beriods after the last date in 7.
    Contains only date, the last buy price, and weekend and holidays designation.
    """
    last_row = df.iloc[[-1]]
    
    from_date = last_row[date_col][0] + timedelta(1)
    # we want `h` new days on top of the last known -> +1
    to_date = from_date + timedelta(h-1)
    
    df_fut = pd.DataFrame(
        pd.date_range(start = from_date, end = to_date),
        columns = [DATE_COL]
        )
    
    df_fut[BUY_PRICE_COL] = last_row[BUY_PRICE_COL][0]
    
    df_fut = is_weekend(df_fut, date_col)
    df_fut = add_czech_holidays(df_fut, date_col, date_format)
    
    df_fut.set_index(date_col, drop=False, inplace=True)
    
    return df_fut


def generate_price_sequence(df, price_col, price_events_col):
    """
    Finds historical max and min price changes and generates a sequence of
    possible prices in between with steps by 2% of current price.
    """
    max_increase = max(df[price_events_col])
    min_increase = min(df[price_events_col])
    
    last_price = df[price_col][-1]
    step = last_price * 0.02
    
    return np.arange(min_increase, max_increase, step)

    
def add_price_and_events(df, df_fut, price_col, margin_col, price_adj, prom_conf):
    """
    Adds price events features into newly generated data for a set `new_price`.
    """
    # last row of the old data contains some needed info
    last_row = df.iloc[[-1]]
    # setting new price event, fast workaround, ok for now
    current_event = last_row["price_event"][0]
    df_fut["price_event"] = current_event + price_adj
    df_fut[price_col] = last_row[price_col][0] + price_adj
    
    df_fut[margin_col] = df_fut[price_col] - df_fut[BUY_PRICE_COL]
    
    return df_fut


    
    



# %% ETL

df_raw = pd.read_csv("ml_task_data.csv")
df_prices = init_transform(df_raw, DATE_COL, DATE_FORMAT, PRICE_COL, MARGIN_COL, ID_COL)

pid_df = split_df(df_prices, ID_COL)

# we need the features to forecast from past ot future
max_dates = {}
for pid in pid_df.keys():
    pid_df[pid] = general_transform(
        pid_df[pid], TRANSFORM_CONF, DATE_COL, PRICE_COL, PROMOTION_CONF
        )
    
    max_dates[pid] = max(pid_df[pid][DATE_COL])
    
    

# %% load the StatsForecst models

with open("exog_cols.json", "r") as fr:
    exog_cols = json.load(fr)

with open("model.pickle", "rb") as infile:
    sf = pickle.load(infile)
    
# %%
    
pid = "0"    
df = pid_df[pid]

abs_margin = estimate_absolute_margin(df, MARGIN_COL, SALES_COL)
df_fut = generate_future_data(df, DATE_COL, DATE_FORMAT)

price_levels = generate_price_sequence(df, PRICE_COL, "price_event")
price_adj = price_levels[10]

df_fut = add_price_and_events(df, df_fut, PRICE_COL, MARGIN_COL, price_adj, PROMOTION_CONF)


# %%

y = df[SALES_COL].values
h = 7
X = df[exog_cols]
X_future = df_fut[exog_cols]

arima = sf.fitted_[0][0]

arima.forward(y, h, X, X_future, level=[95])





    