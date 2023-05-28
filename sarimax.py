import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

WORKDIR = "c:\\Text\\work_search_summer23\\rohlik\\ml_task"

from conf import DATE_FORMAT, TRANSFORM_CONF, DATE_COL, ID_COL, PRICE_COL, MARGIN_COL, SALES_COL
from utils import split_df, general_transform, init_transform



# %% ETL

df_prices = pd.read_csv("ml_task_data.csv")

df_prices, id_map = init_transform(
    df_prices, DATE_COL, DATE_FORMAT, PRICE_COL, MARGIN_COL, ID_COL
    )
pids = list(id_map.values())

pid_df = split_df(df_prices, ID_COL)

for pid in pids:
    pid_df[pid] = general_transform(pid_df[pid], TRANSFORM_CONF)
    
# %% data for testing
pid = "0"
df = pid_df[pid]
# df.set_index(pd.DatetimeIndex(df[DATE_COL]), drop=False, inplace=True)

dff = df[[ID_COL, DATE_COL, SALES_COL]]
dff.columns = ["unique_id", "ds", "y"]
dff_train = dff[:-14]
dff_test =  dff[-14:]  # take last 2 weeks of data

# %%
df_train = df_train.astype({SALES_COL: np.float64})
df_train.dtypes


# %%



sf = StatsForecast(
    models = [AutoARIMA(
        season_length = 7
        )],
    freq = 'D'
)

sf.fit(dff_train)
model_info = sf.fitted_[0][0].model_

# %%

forecast_df = sf.predict(h=14, level=[95]) 
dff_test_pred = dff_test.merge(forecast_df, on=["unique_id", "ds"])
dff_test_pred

# calc MAE, MAPE (both have good interpretation) and RMSE
MAE = mean_absolute_error(dff_test_pred.y, dff_test_pred.AutoARIMA)
MAPE = mean_absolute_percentage_error(dff_test_pred.y, dff_test_pred.AutoARIMA) * 100
RMSE = np.sqrt(mean_squared_error(dff_test_pred.y, dff_test_pred.AutoARIMA))
print("MAE:  ", round(MAE, 2))
print("MAPE: ", round(MAPE, 2), "%")
print("RMSE: ", round(RMSE, 2))
