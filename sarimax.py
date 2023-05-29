import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, SeasonalWindowAverage

WORKDIR = "c:\\Text\\work_search_summer23\\rohlik\\ml_task"

from conf import DATE_FORMAT, TRANSFORM_CONF, DATE_COL, ID_COL, PRICE_COL, MARGIN_COL, SALES_COL
from utils import split_df, general_transform, init_transform


# %%

TEST_SIZE = 14
SEASON_LENGTH = 7

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

exog_cols = [f"{PRICE_COL}_pct_change_1"]
pid_df_train_test = defaultdict(dict)

for pid, df in pid_df.items():
    # StatsForecast requires data in a specific format, that's why we select and rename
    dff = df[[ID_COL, DATE_COL, SALES_COL, *exog_cols]]
    dff.columns = ["unique_id", "ds", "y", *exog_cols]
    pid_df_train_test[pid]["train"] = dff[1:-TEST_SIZE]
    pid_df_train_test[pid]["test"] =  dff[-TEST_SIZE:]
    


# %%

fitted_models = {}

for pid in pid_df.keys():
    sf = StatsForecast(
        models = [
            AutoARIMA(
                max_q=5,
                max_p=5,
                max_d=3,
                max_Q=3,
                max_P=3,
                max_D=1,
                season_length=SEASON_LENGTH),
            # just to check against a simpler method
            SeasonalWindowAverage(season_length=SEASON_LENGTH, window_size=SEASON_LENGTH*2),
            SeasonalNaive(season_length=SEASON_LENGTH) 
            ],
        freq='D',
        n_jobs=-1
    )
    
    sf.fit(pid_df_train_test[pid]["train"])
    fitted_models[pid] = sf

# %%


def predict(sf: StatsForecast, h:int, df_test: pd.DataFrame, exog_cols: list, level:list=[95]): 
    """
    Get prediction of all models in `sf` for the next `h`.
    Uses exogenous variables contained in `df_test` if applicable.
    `level` is a list of confidence intervals to be calculated if applicable.
    """
    X_df = df_test.drop(["y"], 1) if exog_cols else None    
    forecast_df = sf.predict(h=h, X_df=X_df, level=level) 

    return df_test.merge(forecast_df, on=["unique_id", "ds"])

def pick_best_model_metrics(model_metrics: dict):
    """
    Expects input in the form {model: {metric: value}}.
    Finds the model which attains the lowest (!) scores most of the time.
    Returns multiple models in case of ties.
    """
    best_vals = defaultdict(dict)
    
    for model, metrics in model_metrics.items():
        for metric, val in metrics.items():
            curr_best = best_vals.get(metric, {}).get("val", np.inf)
            # check for new best score for this metric
            if val < curr_best:
                best_vals[metric]["val"] = val
                best_vals[metric]["model"] = [model]
            elif val == curr_best:
                # list more models in case of tie, not likely to happoen
                best_vals[metric]["model"] = best_vals[metric].get("model", []) + [model]
                
                
    winners = 
    win_cnts = Counter()
    
 
def eval_predictions(sf: StatsForecast, df_test_pred):
    """
    Basic prediction quality evaluation for all models fitted in `sf`.
    `df_test_pred` should contain predictions for all models present in sf.
    """
    model_names = [str(s) for s in sf.fitted_[0]]

    # calc MAE, MAPE (both have good interpretation) and RMSE
    test_eval_metrics = defaultdict(dict)
    for model in model_names:
        test_eval_metrics[model]["MAE"] = mean_absolute_error(df_test_pred.y, df_test_pred[model])
        test_eval_metrics[model]["MAPE"] = mean_absolute_percentage_error(df_test_pred.y, df_test_pred[model]) * 100
        test_eval_metrics[model]["RMSE"] = np.sqrt(mean_squared_error(df_test_pred.y, df_test_pred[model]))
        
    return test_eval_metrics



# %%

pid_test_eval_metrics

for pid in pid_df.keys():
    # predict sales on test set
    test_prediction = predict(fitted_models[pid], TEST_SIZE, pid_df_train_test[pid]["test"], exog_cols)
    pid_test_eval_metrics[pid] = eval_predictions(fitted_models[pid], test_prediction)
    



# %%

dff_test_pred = predict(sf, dff_test, exog_cols, TEST_SIZE)
models_test_eval_metrics = eval_predictions(sf, dff_test_pred)

# %%


dff_test_pred.set_index("ds", inplace=True)
dff_test_pred[['y', "AutoARIMA"]].plot()

# %%
plot_cols = ["y", "ds", "AutoARIMA", "AutoARIMA-lo-95", "AutoARIMA-hi-95"]

# surprisingly slow, investigate better method to obtain the data
f_df = sf.forecast(h=14, X_df=X_df, fitted=True, level=[95]) 
insample_fcsts_df = sf.forecast_fitted_values()

insample_fcsts_df = insample_fcsts_df[-50:]

df_res_vis = pd.concat([insample_fcsts_df[plot_cols], dff_test_pred[plot_cols]])

df_res_vis.rename({"y": "sales", "AutoARIMA": "estimate", "ds": "date"}, axis=1, inplace=True)
df_res_vis.set_index("date", inplace=True)

df_res_vis[["sales", "estimate"]].plot(linewidth=.5, alpha=.7)
plt.fill_between(df_res_vis.index, df_res_vis["AutoARIMA-lo-95"], df_res_vis["AutoARIMA-hi-95"], alpha=.2)
plt.axvline(x=insample_fcsts_df.ds[-1], linewidth=.3)
plt.title("fitted and predicted values")
plt.savefig("test_predict.png", dpi=500)




