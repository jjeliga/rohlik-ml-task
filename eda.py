# In the link below, you can find time series data for 5 randomly chosen products that we sell in Rohlik.
# Your task is to explore these time series. Find what is the best price for which we can sell each product
# in next 7 days (starting from the last day of time series). Goal is to maximize revenue (sales * sell_price)
# while maintaining absolute margin - (sell_price - buy_price) * sales. For margin calculation, please count
# that buy price for next 7 days will be same as it was in the last day of given time series. You can
# enhance this data set with some external data, it’s up to you, be creative :-).

# download and unzip ml_task_data prior to running this script
# https://temp-roh.s3.eu-central-1.amazonaws.com/ml_task_data.zip

import os
import warnings
import json
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

from conf import (
    DATE_FORMAT,
    TRANSFORM_CONF,
    PROMOTION_CONF,
    ID_COL,
    DATE_COL,
    MARGIN_COL,
    SALES_COL,
    PRICE_COL
   )
from utils import split_df, general_transform, init_transform

# %% basic conf

os.makedirs("plots", exist_ok=True)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

# saves generated figures to plots folder if True, else show
SAVE_FIGURES = True

# %% read data and peek

df_raw = pd.read_csv("ml_task_data.csv")
print(df_raw.head())

# %% check NAs
df_raw.info()

# %% check basic stats
print(df_raw.describe())

# %% initial transdformation
# convert date col to datetime format and map ids for better readability

df_prices, id_map = init_transform(df_raw, DATE_COL, DATE_FORMAT, PRICE_COL, MARGIN_COL, ID_COL)
with open("id_map.json", "w") as fw:
    json.dump(id_map, fw)

# %% split to individual dfs, necessary for correct pct_change calculation
pid_df = split_df(df_prices, ID_COL)


# %% make some transformations and create feature columns

for pid in pid_df.keys():
    pid_df[pid] = general_transform(
        pid_df[pid], TRANSFORM_CONF, DATE_COL, PRICE_COL, PROMOTION_CONF
        )


# %% check basic statistics per product
stat_cols = set(df_prices.columns) - {ID_COL}

for pid, df in pid_df.items():
    print()
    print(pid)
    print(df[stat_cols].agg(["min", "median", "mean", "max", "count", "std"]))


# %% check dates span per product
df_dates_ranges = df_prices.groupby(ID_COL)[DATE_COL].agg(["min", "max"])
print(df_dates_ranges)

# %% find missing dates per product
missing_dates = {}
for pid, df in pid_df.items():
    dmin, dmax = df_dates_ranges.loc["0"]
    missing_dates[pid] = pd.date_range(start = dmin, end = dmax).difference(df[DATE_COL])
    
print(missing_dates)

# %% missing dates 1
# visualisation of missing dates
for pid, missing in missing_dates.items():
    dmin, dmax = df_dates_ranges.loc[pid]
    df_full_dates = pd.DataFrame(pd.date_range(start = dmin, end = dmax), columns = [DATE_COL])
    df_pid_miss_dates = pd.DataFrame(missing, columns = [DATE_COL])
    df_pid_miss_dates["miss"] = 1
    df_full_dates = df_full_dates.merge(df_pid_miss_dates, how="left", on=DATE_COL)
    df_full_dates = df_full_dates.fillna(0)
    plt.plot(df_full_dates.date, df_full_dates.miss)
    plt.title(f"missing dates for product {pid}")
    if SAVE_FIGURES:
        plt.savefig(f"plots/{pid}_miss_dates.png")
    else:
        plt.show()
    plt.close()

# %% missing dates 2

for pid, df in pid_df.items():
    tdeltas = [d.days - 1 for d in df[DATE_COL] - df[DATE_COL].shift(1)]
    tdeltas_miss = [t for t in tdeltas if t > 0]
    plt.hist(tdeltas_miss, bins=20)
    plt.title(f"frequencies of n consequent missing days for product {pid}")
    plt.xlabel("n consequent days missing")
    if SAVE_FIGURES:
        plt.savefig(f"plots/{pid}_dates_gaps_frequencies.png")
    else:
        plt.show()
    plt.close()

# Upon inspecting the plots.
# Lets just not care about the missing values for now.
# ARIMA estimation procedure can handle missing observations and inappropriate
# imputation might distort the estimates.
# Big influence on the final result is not expected since the missing blocks
# Are mostly of very short sizes.


# %% exploring relation of sales and other varibales

# there is a visible spike in sales after a price drop which lasts 
# otherwise looks rather noisy
# visible changes after the beginning of covid pandemia, but the not very significant

for pid, df in pid_df.items():
    # creating and saving the plot of sales and other variables
    fig, ax = plt.subplots()
    lns1 = ax.plot(df[DATE_COL], df[PRICE_COL], linewidth=.5, alpha=.7, label="price")
    lns2 = ax.plot(df[DATE_COL], df[MARGIN_COL], linewidth=.5, alpha=.7, label="margin")
                   
    
    ax2 = ax.twinx()
    lns3 = ax2.plot(df[DATE_COL], df[SALES_COL], color="green", linewidth=.5, alpha=.5, label="sales")
    # TODO: make separate axis for sales
    
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.title(f"sales and related variables of product {pid}")
    # ax.legend([PRICE_COL, MARGIN_COL, SALES_COL])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if SAVE_FIGURES:
        plt.savefig(f"plots/{pid}_sales_related.png", dpi=500)
    else:
        plt.show()
    plt.close()

# %% exploring price events 1
# significant change of price will be used as an exogenous variable in sarimax model


sale_change_col = f"{SALES_COL}_diff_1"
price_change_col = f"{PRICE_COL}_diff_1"

# possible to try different perspective, not very useful
# sale_change_col = f"{SALES_COL}_pct_change_1"
# pricee_change_col = f"{PRICE_COL}_pct_change_1"

# exploring only immediate influence
for pid, df in pid_df.items():
    # take only noticeable price changes
    df_changes = df[abs(df.sell_price_pct_change_1) > 0.1]
    # exploring the relation of relative price change and absolute sales change 
    plt.scatter(df_changes[price_change_col], df_changes[sale_change_col])
    plt.xlabel(price_change_col)
    plt.ylabel(sale_change_col)
    plt.title(f"price dynamics of product {pid}")

    if SAVE_FIGURES:
        plt.savefig(f"plots/{pid}_price_change_dynamics.png", dpi=500)
    else:
        plt.show()
    plt.close()
    
# After inspecting the resulting figures, it seems that there is a more or less
# linear relationship between price difference and sales difference
# for a more significant changes in price.
# Might be caused by a constant pool size of price sensitive individuals.

# %% exploring price events 2
# looking at prices and price events simultaneously
for pid, df in pid_df.items():
    fig, ax = plt.subplots()
    # df[[f"{PRICE_COL}_pct_change_1", "price_event_pct"]].plot(title=f"promotions of product {pid}")
    df[[f"{PRICE_COL}", "price_event"]].plot(title=f"price events of product {pid}", ax=ax)
    if SAVE_FIGURES:
        fig.savefig(f"plots/{pid}_price_events.png", dpi=500)
    else:
        plt.show()
    plt.close()

# %% stationarity tests

# the results indicate that sales ts for products 0-3 might be (trend-)stationary
# when looking at plots of their sales, they seem to be little more than a WN
# exploring out of curiosity, does not have influence on our sales modeling experimets

for pid, df in pid_df.items():
    # testing for trend-stationarity
    result = adfuller(df[SALES_COL], regression="ct")
    print(pid)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))

# %% autocorelatinon

for pid, df in pid_df.items():
    plot_acf(df[SALES_COL])
    plt.title(f"Autocorrelation of product {pid}")

    if SAVE_FIGURES:
        plt.savefig(f"plots/{pid}_sales_autocorr.png", dpi=500)
    else:
        plt.show()
    plt.close()

# we can observe significant autocorrelation of at least 5th order for each product
# results might be uninformative for non-stationary processes
# exploring out of curiosity, does not have influence on our sales modeling experimets


# %% partial autocorelatinon

for pid, df in pid_df.items():
    plot_pacf(df[SALES_COL])
    plt.title(f"Partial Autocorrelation of product {pid}")
    
    if SAVE_FIGURES:
        plt.savefig(f"plots/{pid}_sales_pautocorr.png", dpi=500)
    else:
        plt.show()
    plt.close()
    
# partial autocorrelations fade rather quickly for most, usualy within lag of 5
# gives some estimate of the order of the AR part of the sales "processes"
# results might be uninformative for non-stationary processes










