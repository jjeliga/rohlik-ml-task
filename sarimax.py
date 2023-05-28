import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

exog_cols = [f"{PRICE_COL}_pct_change_1"]

dff = df[[ID_COL, DATE_COL, SALES_COL, *exog_cols]]
dff.columns = ["unique_id", "ds", "y", *exog_cols]
dff_train = dff[1:-14]
dff_test =  dff[-14:]  # take last 2 weeks of data


# %%



sf = StatsForecast(
    models = [AutoARIMA(
        max_q=5,
        max_p=5,
        max_d=3,
        max_Q=3,
        max_P=3,
        max_D=1,
        season_length=7
        )],
    freq='D',
    # n_jobs=-1
)

sf.fit(dff_train)
model_info = sf.fitted_[0][0].model_

# %%

X_df = dff_test.drop(["y"], 1) if exog_cols else None

forecast_df = sf.predict(h=14, X_df=X_df, level=[95]) 
dff_test_pred = dff_test.merge(forecast_df, on=["unique_id", "ds"])
dff_test_pred

# calc MAE, MAPE (both have good interpretation) and RMSE
MAE = mean_absolute_error(dff_test_pred.y, dff_test_pred.AutoARIMA)
MAPE = mean_absolute_percentage_error(dff_test_pred.y, dff_test_pred.AutoARIMA) * 100
RMSE = np.sqrt(mean_squared_error(dff_test_pred.y, dff_test_pred.AutoARIMA))
print("MAE:  ", round(MAE, 2))
print("MAPE: ", round(MAPE, 2), "%")
print("RMSE: ", round(RMSE, 2))



# %%


dff_test_pred.set_index("ds", inplace=True)
dff_test_pred[['y', "AutoARIMA"]].plot()

# %%
plot_cols = ["y", "ds", "AutoARIMA", "AutoARIMA-lo-95", "AutoARIMA-hi-95"]

f_df = sf.forecast(h=14, X_df=X_df, fitted=True, level=[95]) 
insample_fcsts_df = sf.forecast_fitted_values()

df_res_vis = pd.concat([insample_fcsts_df[plot_cols], dff_test_pred[plot_cols]])
df_res_vis.set_index("ds", inplace=True)

df_res_vis.plot(linewidth=.5)
plt.axvline(x=insample_fcsts_df.ds[-1], linewidth=.3)
plt.savefig("test_predict.png", dpi=500)




