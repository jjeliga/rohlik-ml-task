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

os.chdir("c:\\Text\\work_search_summer23\\rohlik\\task")
df_prices = pd.read_csv("ml_task_data.csv")

# basic info
df_prices.head()
df_prices.info() # check NAs
df_prices.describe()

# convert date col to datetime format
df_prices.date = pd.to_datetime(df_prices.date, format="%Y-%m-%d")

# rename ids
old_pids = set(df_prices.product_id)
pids = [str(i) for i in range(len(old_pids))]
id_map = dict(zip(old_pids, pids))

df_prices.product_id = df_prices.product_id.map(id_map)
df_prices.head()

# sort for better navigation, cheap since the data is small
df_prices = df_prices.sort_values(["product_id","date"])

# check basic statistics per product
for col in ["sell_price", "margin", "sales"]:
    print(df_prices.groupby("product_id")[[col]].agg(["min", "median", "mean", "max", "count", "std"]))

# check dates span
df_dates_ranges = df_prices.groupby("product_id")[["date"]].agg(["min", "max"])
print(df_dates_ranges)

missing_dates = {}
for pid in pids:
    df_pid = df_prices[df_prices.product_id == pid]
    dmin, dmax = df_dates_ranges.loc["0"]
    missing_dates[pid] = pd.date_range(start = dmin, end = dmax).difference(df_pid.date)
    
print(missing_dates)
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

# there is a visible spike in sales after a price drop, otherwise some noise + trend
for pid in pids:
    df_pid = df_prices[df_prices.product_id == pid]
    plt.plot(df_pid.date, df_pid.sell_price, df_pid.date, df_pid.sales, df_pid.date, df_pid.margin, linewidth=1)
    plt.title(f"price, sales and margin of product {pid}")
    plt.legend(["sell_price", "sales", "margin"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{pid}_price_sales_margin.png", dpi=500)
    plt.close()


















