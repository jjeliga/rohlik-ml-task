import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    SeasonalNaive,
    SeasonalWindowAverage,
    SeasonalExponentialSmoothingOptimized,
    HoltWinters,
    
    
)
WORKDIR = "c:\\Text\\work_search_summer23\\rohlik\\ml_task"

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


# %%

TEST_SIZE = 14
SEASON_LENGTH = 7

# %%


def predict(sf: StatsForecast, df_train: pd.DataFrame, df_test: pd.DataFrame, exog_cols: list, level:list=[95]): 
    """
    Get prediction of all models in `sf` for all dates in df_test and fitted values.
    Uses exogenous variables contained in `df_test` if applicable.
    `level` is a list of confidence intervals to be calculated if applicable.
    """
    h = df_test.shape[0]
    X_df = df_test[["unique_id", "ds", *exog_cols]] if exog_cols else None    
    forecast_df = sf.forecast(h=h, df=df_train, X_df=X_df, level=level, fitted=True) 
    df_p = df_test.merge(forecast_df, on=["unique_id", "ds"])
    
    fitted = sf.forecast_fitted_values()

    return fitted, df_p


def pick_best_model(models_metrics: dict, eval_metrics: list):
    """
    Finds the model which attains the lowest (!) scores most of the time.
    Expects input data in the form {model: {metric: value}} and list of metrics
    names considered for model selection.
    Returns multiple models in case of ties in no of wins.
    """
    best_vals = defaultdict(dict)
    
    for model, metrics in models_metrics.items():
        for metric, val in metrics.items():
            if metric not in eval_metrics:
                continue
            # not very elegant but safe and readable approach
            curr_best = best_vals.get(metric, {}).get("val", np.inf)
            # check for new best score for this metric
            if val < curr_best:
                best_vals[metric]["val"] = val
                best_vals[metric]["model"] = [model]
            elif val == curr_best:
                # list more models in case of tie, not likely to happoen
                best_vals[metric]["model"] = best_vals[metric].get("model", []) + [model]
                
    # count wins
    winners_per_metric = [Counter(v["model"]) for v in best_vals.values()]
    winners = sum(winners_per_metric, Counter())
    
    # select best model
    best_models, best_n_wins = [], 0
    for model, n_wins in winners.items():
        if n_wins > best_n_wins:
            best_n_wins = n_wins
            best_models = [model]
        elif n_wins == best_n_wins:
            best_models.append(model)
            
    return best_models



def _eval_predictions_single(sf: StatsForecast, df_pred: pd.DataFrame):
    """
    Basic prediction quality evaluation for all models fitted in `sf`.
    `df_pred` should contain predictions for all models present in sf.
    """
    model_names = [str(m) for m in sf.models]

    # Calculate absolute total sum difference (ATSD),  MAE, MAPE (both have good interpretation) and RMSE.
    # ATSD is a custom metric, might be useful for a setting where only
    # the absolute volume over a longer period of time is of interest
    
    model_eval_metrics = defaultdict(dict)
    for model in model_names:
        model_eval_metrics[model]["ATSD"] = abs(df_pred.y.sum() - df_pred[model].sum())
        model_eval_metrics[model]["MAE"] = mean_absolute_error(df_pred.y, df_pred[model])
        model_eval_metrics[model]["MAPE"] = mean_absolute_percentage_error(df_pred.y, df_pred[model]) * 100
        model_eval_metrics[model]["RMSE"] = np.sqrt(mean_squared_error(df_pred.y, df_pred[model]))
        
    return model_eval_metrics


def eval_predictions(fitted_models: dict, predictions: dict):
    pid_eval_metrics = {}
    for pid, prediction in predictions.items():
        pid_eval_metrics[pid] = _eval_predictions_single(fitted_models[pid], prediction.dropna())
        
    # quick overview
    for pid, metrics in pid_eval_metrics.items():
        print(pid)
        for mod, met in metrics.items():
            print(mod)
            print(met)
            
    return pid_eval_metrics

# %% ETL

df_prices = pd.read_csv("ml_task_data.csv")

df_prices = init_transform(
    df_prices, DATE_COL, DATE_FORMAT, PRICE_COL, MARGIN_COL, ID_COL
    )

pid_df = split_df(df_prices, ID_COL)

for pid in pid_df.keys():
    pid_df[pid] = general_transform(
        pid_df[pid], TRANSFORM_CONF, DATE_COL, PRICE_COL, PROMOTION_CONF
        )
    
# %% data for testing

exog_cols = ["is_weekend", "holiday", "is_promotion"]  # f"{PRICE_COL}_pct_change_1" ommited for now
pid_df_train_test = defaultdict(dict)

for pid, df in pid_df.items():
    # StatsForecast requires data in a specific format, that's why we select and rename
    dff = df[[ID_COL, DATE_COL, SALES_COL, *exog_cols]]
    dff.columns = ["unique_id", "ds", "y", *exog_cols]
    # drop to account for the lagged variables
    pid_df_train_test[pid]["train"] = dff[:-TEST_SIZE].dropna() 
    pid_df_train_test[pid]["test"] =  dff[-TEST_SIZE:]
    


# %% fit various ts models to train data

fitted_models = {}

for pid in pid_df.keys():
    sf = StatsForecast(
        models = [
            AutoARIMA(
                max_q=7,  # MA order
                max_p=7,  # AR order
                max_d=2,  # difference order
                max_Q=1,
                max_P=1,
                max_D=1,
                start_p=0, start_q=0,
                start_P=0, start_Q=0,
                season_length=SEASON_LENGTH,
                # nmodels=1000,
                ),
            SeasonalExponentialSmoothingOptimized(season_length=SEASON_LENGTH),
            HoltWinters(season_length=SEASON_LENGTH),
            # Checking against some simpler benchmark methods.
            # These methods do not account for price changes, therefore we will not
            # try to use those for an estimation of sales with respect to price
            # when trying to maximize profit. The solution would be to create a separate
            # price elasticity model and then try to combine these.            
          #  SeasonalWindowAverage(season_length=SEASON_LENGTH, window_size=SEASON_LENGTH*2),
            SeasonalNaive(season_length=SEASON_LENGTH) 
            ],
        freq='D',
        n_jobs=-1
    )
    
    sf.fit(pid_df_train_test[pid]["train"])
    fitted_models[pid] = sf
# %% look at arima coefficients

fitted_models["0"].fitted_[0][0].model_["coef"]


# %%

# from prophet import Prophet
# from workalendar.europe import CzechRepublic

# pid="1"
# dfte = pid_df_train_test[pid]["test"]
# dftr = pid_df_train_test[pid]["train"]



# m = Prophet(holidays=holidays)
# m.add_regressor("sell_price_pct_change_1")
# m.fit(dftr[["ds", "y", "sell_price_pct_change_1"]])

# forecast = m.predict(dfte[["ds", "sell_price_pct_change_1"]])
# forecast.yhat
# dfte.y

# dfte.iloc[7, -1] = -0.2
# dfte.sell_price_pct_change_1

# forecast = m.predict(dfte[["ds", "sell_price_pct_change_1"]])
# forecast.yhat
# dfte.y




# %% get predictions on test set and fitted values

fitted_values = {}
test_predictions = {}

for pid in pid_df.keys():
    # obtain fitted values and predict sales on test set
    dftr = pid_df_train_test[pid]["train"].copy()
    dfte = pid_df_train_test[pid]["test"].copy()
    fv, tp = predict(fitted_models[pid], dftr, dfte, exog_cols)
    fitted_values[pid] = fv
    test_predictions[pid] = tp

# %% evaluate models fit

# AutoARIMA with exogenous variables attains the best scores/fit according to our metrics
fit_eval_metrics = eval_predictions(fitted_models, fitted_values)


# %% evaluate models on test data

# It seems that the very best out-of-sample performing models are usually the most simple ones.
# However, only the ARIMA model directly incorporates the influence of exogenous variables so
# we do not need to combine the sales forecast with another methods using price elasticity
# of demand which would be neded for revenue optimisation

test_eval_metrics = eval_predictions(fitted_models, test_predictions)


# %%
# these metrics seem to be most directly correlated with business impact

# selection_metrics = ["ATSD", "MAE"]
# best_models = {
#     pid: pick_best_model(mm, selection_metrics) for pid, mm in pid_test_eval_metrics.items()
#     }
# print(best_models)


# %%

dff_test_pred.set_index("ds", inplace=True)
dff_test_pred[['y', "AutoARIMA"]].plot()

# %%


model_names = [str(m) for m in sf.models]

plot_cols = ["y", "ds", *model_names, "AutoARIMA-lo-95", "AutoARIMA-hi-95"]

# surprisingly slow, investigate better method to obtain the data

insample_fcsts_df = [-50:]

df_res_vis = pd.concat([insample_fcsts_df[plot_cols], df_test_pred[plot_cols]])

df_res_vis.rename({"y": "sales", "AutoARIMA": "estimate", "ds": "date"}, axis=1, inplace=True)
df_res_vis.set_index("date", inplace=True)

df_res_vis[["sales", "estimate"]].plot(linewidth=.5, alpha=.7)
plt.fill_between(df_res_vis.index, df_res_vis["AutoARIMA-lo-95"], df_res_vis["AutoARIMA-hi-95"], alpha=.2)
plt.axvline(x=insample_fcsts_df.ds[-1], linewidth=.3)
plt.title("fitted and predicted values")
plt.savefig("test_predict.png", dpi=500)





