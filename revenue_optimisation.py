import pickle
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from itertools import product

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
    add_czech_holidays
    )

# %% script conf

SELECTED_MODEL = "AutoARIMA"
FORECAST_HORIZON = 7

# %% helper functions

def estimate_absolute_margin(df, margin_col, sales_col, window=7):
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


def generate_price_adj_sequence(df, price_col, margin_col, price_events_col):
    """
    Finds historical max price change and current margin level and generates a sequence of
    possible prices in between with steps by 2% of current price.
    We do not allow for a negative margin.
    """
    max_change = max(df[price_events_col])
    min_change = -df[margin_col][-1]
    
    last_price = df[price_col][-1]
    # trying changes of .5% magnitude of the current price
    step = last_price * 0.005
    
    seq = np.arange(min_change, max_change, step)
    # include no change
    seq = np.append(seq, [0])
    return seq

    
def add_price_and_events(df, df_fut, price_col, margin_col, price_adj):
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

def estimate_revenue_margin(df, df_fut, model, sales_col, price_col, buy_price_col, exog_cols, h=7):
    """
    Estimates total revenue and margin in `h` days forecast horizon using the 
    `model` for sales volume estimation based on price change scenario data in 
    `df_fut`.
    """
    # statistical confidence levels for the sales prediction
    levels = [50, 75]
    y = df[sales_col].values
    X = df[exog_cols].to_numpy()
    X_future = df_fut[exog_cols].to_numpy()
    
    # using fitted model to pass through the available data and generate predictions
    predictions = model.forward(y, h, X, X_future, level=levels)
    
    price = df_fut[price_col].values[0]
    margin = df_fut[price_col].values[0] - df_fut[buy_price_col].values[0]

    info = {}
    # computing point estrimates
    info["price"] = price
    info["sales"] = sum(predictions["mean"])
    info["revenue"] = price * info["sales"]
    info["margin"] = margin * info["sales"]

    # confidence intervals for sales, revenue and margin
    # aggregated values, so the intervals are jsut approximate
    for hl, lev in product(["hi", "lo"], levels):
        info[f"sales_{hl}_{lev}"] = sum(predictions[f"{hl}-{lev}"])
        info[f"revenue_{hl}_{lev}"] = price * info[f"sales_{hl}_{lev}"]
        info[f"margin_{hl}_{lev}"] = margin * info[f"sales_{hl}_{lev}"]
    
    return info

def optimise_revenue(df, model, price_col, buy_price_col, margin_col, sales_col, date_col, date_format, h=7):
    """
    Function to generate sales, revenue and margin estimates using the `model`
    for sales estimation. Prediction is performed on `h` days forecast horizon.
    The different scenarios are determined by selected price changes.
    """
    # estimating current absolute margin level
    abs_margin = estimate_absolute_margin(df, margin_col, sales_col, h)
    df_fut = generate_future_data(df, date_col, date_format)
    # generating price change options for different scenarios
    price_adjustments = generate_price_adj_sequence(df, price_col, margin_col, "price_event")
    
    # estimates for all scenarios
    estimates = []
    # estimates for scenarios with margin >= abs_margin
    winning_estimates = []
    for price_adj in price_adjustments:
        # transforming the "future" data according to the new price adjustment
        df_fut = add_price_and_events(df, df_fut, price_col, margin_col, price_adj)
        # generating the estimates
        rev_estimates_info = estimate_revenue_margin(
            df, df_fut, model, sales_col, price_col, buy_price_col, exog_cols, h)
        rev_estimates_info["adjustment"] = price_adj
        estimates.append(rev_estimates_info)
        
        if rev_estimates_info["margin"] >= abs_margin:
            winning_estimates.append(rev_estimates_info)
            
    return estimates, winning_estimates

def pick_winning_scenario(estimates, winning_estimates):
    """
    Finding the best price adjustment scenario in given scenario results.
    """
    if winning_estimates:
        # if there are scenarios with high enough margin
        # we choose the one with highest revenue
        return max(winning_estimates, key=lambda x: x['revenue']), True
    else:
        # in case that no scenario has high enough margin we take the one
        # which maximizes margoin and consequently minimizes losses
        return max(estimates, key=lambda x: x['margin']), False


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
    
    

# %% load the StatsForecst models and exogenous col names

with open("exog_cols.json", "r") as fr:
    exog_cols = json.load(fr)

with open("models.pickle", "rb") as infile:
    models = pickle.load(infile)
    
    
# %% generating sales, revenue and price estimates for all products

optimal_scenarios = {}
for pid, df in pid_df.items():
    # retrieve saved mdoel
    sf = models[pid]    
    model_names = [str(m) for m in sf.models]
    sel_idx = model_names.index(SELECTED_MODEL)
    model = sf.fitted_[0][sel_idx]
    
    # generate the estimates, obtaining aggregated results
    estimates, w_estimates = optimise_revenue(
        df, model, PRICE_COL, BUY_PRICE_COL, MARGIN_COL, SALES_COL, DATE_COL, DATE_FORMAT, FORECAST_HORIZON
        )
    
    optimal_scenario, is_better = pick_winning_scenario(estimates, w_estimates)
    print(f"optimal scenario results for product {pid} is:")
    print(optimal_scenario)
    
    optimal_scenarios[pid] = {"scenario": optimal_scenario, "better_than_recent": is_better}

# %% saving the results

with open("optimal_scenarios.json", "w") as fw:
    json.dump(optimal_scenarios, fw)
  
 
    
        