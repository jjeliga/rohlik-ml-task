# In the link below, you can find time series data for 5 randomly chosen products that we sell in Rohlik.
# Your task is to explore these time series. Find what is the best price for which we can sell each product
# in next 7 days (starting from the last day of time series). Goal is to maximize revenue (sales * sell_price)
# while maintaining absolute margin - (sell_price - buy_price) * sales. For margin calculation, please count
# that buy price for next 7 days will be same as it was in the last day of given time series. You can
# enhance this data set with some external data, it’s up to you, be creative :-).

# download and unzip ml_task_data prior to running this script
# https://temp-roh.s3.eu-central-1.amazonaws.com/ml_task_data.zip


import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', None)

# if __name__=="__main__":
#     print(os.getcwd())
#     print(os.path.dirname(os.path.realpath(__file__)))


os.chdir("c:\\Text\\work_search_summer23\\rohlik\\ml_task")
df_prices = pd.read_csv("ml_task_data.csv")

# %%
# basic info
print(df_prices.head())

# %%
df_prices.info() # check NAs

# %%
df_prices.describe()

# %%
# convert date col to datetime format
df_prices.date = pd.to_datetime(df_prices.date, format="%Y-%m-%d")

# rename ids
old_pids = set(df_prices.product_id)
pids = [str(i) for i in range(len(old_pids))]
id_map = dict(zip(old_pids, pids))

df_prices.product_id = df_prices.product_id.map(id_map)

# %% split to individual dfs, necessary for correct pct_change calculation
pid_df = {}
for pid in pids:
    pid_df[pid] = (
        df_prices
        .query(f"product_id == '{pid}'")
        .sort_values("date")
        )


# %% add percent price change

for pid in pids:
    pid_df[pid]["sell_price_pct_change"] = pid_df[pid]["sell_price"].pct_change(1)


# %%
# check basic statistics per product
cols = ["sell_price", "margin", "sales", "sell_price_pct_change"]

for pid, df in pid_df.items():
    print(pid)
    print(df[cols].agg(["min", "median", "mean", "max", "count", "std"]))


# %%
# check dates span
df_dates_ranges = df_prices.groupby("product_id")[["date"]].agg(["min", "max"])
print(df_dates_ranges)

# %%
missing_dates = {}
for pid, df in pid_df.items():
    dmin, dmax = df_dates_ranges.loc["0"]
    missing_dates[pid] = pd.date_range(start = dmin, end = dmax).difference(df.date)
    
print(missing_dates)

# %%
# visualisation of missing dates
for pid, missing in missing_dates.items():
    dmin, dmax = df_dates_ranges.loc[pid]
    df_full_dates = pd.DataFrame(pd.date_range(start = dmin, end = dmax), columns = ["date"])
    df_pid_miss_dates = pd.DataFrame(missing, columns = ["date"])
    df_pid_miss_dates["miss"] = 1
    df_full_dates = df_full_dates.merge(df_pid_miss_dates, how="left", on="date")
    df_full_dates = df_full_dates.fillna(0)
    plt.plot(df_full_dates.date, df_full_dates.miss)
    plt.title(f"missing dates for product {pid}")
    plt.savefig(f"{pid}_miss_dates.png")
    plt.close()

# lets just not care about the missing values for now
# we can try dropping or performing ffill/interpolation and then compare, 
# the thing is, that this might be basically equivalent

# %% exploring relation of sales and other varibales
# there is a visible spike in sales after a price drop which lasts 
# -> create a sale indicator
# otherwise some noise + trend

for pid, df in pid_df.items():
    # only for visualisation purposes of price changes, no interpretation
    df["pct_change_scaled"] = (df["sell_price_pct_change"] * 
        df["sell_price"].min() / df["sell_price_pct_change"].max())
    # creating and saving the plot of sales and other variables
    plt.plot(
        df.date, df.sell_price,
        df.date, df.sales,
        df.date, df.margin,
        df.date, df.pct_change_scaled,
        linewidth=1)
    plt.title(f"sales and related variables of product {pid}")
    plt.legend(["sell_price", "sales", "margin", "pct_change_scaled"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{pid}_sales_related.png", dpi=500)
    plt.close()


















