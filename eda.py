# In the link below, you can find time series data for 5 randomly chosen products that we sell in Rohlik.
# Your task is to explore these time series. Find what is the best price for which we can sell each product
# in next 7 days (starting from the last day of time series). Goal is to maximize revenue (sales * sell_price)
# while maintaining absolute margin - (sell_price - buy_price) * sales. For margin calculation, please count
# that buy price for next 7 days will be same as it was in the last day of given time series. You can
# enhance this data set with some external data, it’s up to you, be creative :-).

# download and unzip ml_task_data prior to running this script
# https://temp-roh.s3.eu-central-1.amazonaws.com/ml_task_data.zip


import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', None)

WORKDIR = "c:\\Text\\work_search_summer23\\rohlik\\ml_task"

# %%

os.chdir(WORKDIR)

# importing after the change of working directory
from conf import DATE_FORMAT, TRANSFORM_CONF, ID_COL, DATE_COL, MARGIN_COL, SALES_COL, PRICE_COL
from utils import split_df, general_transform, init_transform

# %% read data and peek

df_prices = pd.read_csv("ml_task_data.csv")
print(df_prices.head())

# %%
df_prices.info() # check NAs

# %%
df_prices.describe()

# %%
# convert date col to datetime format and map ids for better readability

df_prices, id_map = init_transform(df_prices, DATE_COL, DATE_FORMAT, PRICE_COL, MARGIN_COL, ID_COL)
pids = list(id_map.values())

# %% split to individual dfs, necessary for correct pct_change calculation
pid_df = split_df(df_prices, ID_COL)


# %% add percent price change

for pid in pids:
    pid_df[pid] = general_transform(pid_df[pid], TRANSFORM_CONF)


# %%
# check basic statistics per product
stat_cols = set(df_prices.columns) - {ID_COL}

for pid, df in pid_df.items():
    print(pid)
    print(df[stat_cols].agg(["min", "median", "mean", "max", "count", "std"]))


# %%
# check dates span
df_dates_ranges = df_prices.groupby(ID_COL)[DATE_COL].agg(["min", "max"])
print(df_dates_ranges)

# %%
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
    plt.savefig(f"{pid}_miss_dates.png")
    plt.close()

# %% missing dates 2

for pid, df in pid_df.items():
    tdeltas = [d.days - 1 for d in df[DATE_COL] - df[DATE_COL].shift(1)]
    tdeltas_miss = [t for t in tdeltas if t > 0]
    plt.hist(tdeltas_miss, bins=20)
    plt.title(f"frequencies of missing days in row for product {pid}")
    plt.show()
    plt.savefig(f"{pid}_dates_gaps_frequencies.png")
    plt.close()

# Lets just not care about the missing values for now.
# ARIMA estimation procedure can handle missing observations and inappropriate
# imputation might distort the estimates.
# Big influence on the final result is not expected since the missing blocks
# Are mostly of shorter sizes.


# %% exploring relation of sales and other varibales

# there is a visible spike in sales after a price drop which lasts 
# otherwise some noise + trend

for pid, df in pid_df.items():
    # only for visualisation purposes of price changes, no interpretation
    df["pct_change_scaled"] = (df[f"{PRICE_COL}_pct_change_1"] * 
        df[PRICE_COL].min() / df[f"{PRICE_COL}_pct_change_1"].max())
    # creating and saving the plot of sales and other variables
    plt.plot(
        df[DATE_COL], df[PRICE_COL],
        df[DATE_COL], df[SALES_COL],
        df[DATE_COL], df[f"{SALES_COL}_diff_1"],
        df[DATE_COL], df[MARGIN_COL],
        df[DATE_COL], df["pct_change_scaled"],
        linewidth=1)
    
    # TODO: make separate axis for sales

    plt.title(f"sales and related variables of product {pid}")
    plt.legend([PRICE_COL, SALES_COL, f"{SALES_COL}_diff_1", MARGIN_COL, f"{PRICE_COL}_pct_change_scaled"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{pid}_sales_related.png", dpi=500)
    plt.close()

# %% exploring the sales
# will be used as an exogenous variable in our model
# the baseline model will be sarimax

for pid, df in pid_df.items():
    # take only noticeable price changes
    df_changes = df[abs(df.sell_price_pct_change_1) > 0.01]
    # exploring the relation of relative price change and absolute sales change 
    plt.scatter(df_changes[f"{PRICE_COL}_pct_change_1"], df_changes.[f"{SALES_COL}_pct_change_1"])
    plt.xlabel("price pct change")
    plt.ylabel("sales pct change")
    plt.title(f"price dynamics of product {pid}")
    plt.savefig(f"{pid}_price_change_dynamics.png", dpi=500)
    plt.close()
    
# after inspecting the resulting figures, it seems that there is a linear 
# or quadratic relationship, we will include both terms in the sarimax model
    
# %% decomposition

from statsmodels.tsa.seasonal import seasonal_decompose, STL

# for pid, df in pid_df.items():
df.set_index(pd.DatetimeIndex(df[DATE_COL], freq="D"), drop=False, inplace=True)
result = seasonal_decompose(df[SALES_COL], model='additive')
fig = result.plot()


# STL


# %% stationarity tests

# the results indicate that sales ts for products 0-3 might be (trend-)stationary

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
    # plt.s1how()
    plt.savefig(f"{pid}_sales_autocorr.png", dpi=500)

# we can observe significant autocorrelation of at least 5th order for each product
# this would indicate that the MA part of used SARIMAX might be quite long


# %% partial autocorelatinon

for pid, df in pid_df.items():
    plot_pacf(df[SALES_COL])
    plt.title(f"Partial Autocorrelation of product {pid}")
    # plt.s1how()
    plt.savefig(f"{pid}_sales_pautocorr.png", dpi=500)
    
# partial autocorrelations fade rather quickly for most, usualy within lag of 5
# giv3es some estimate of the ordr of the AR part










