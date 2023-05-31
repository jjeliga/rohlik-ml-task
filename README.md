# rohlik-ml-task

This repository contains a possible solution for solving revenue optimisation problem stated as follows:

>"In the link below, you can find time series data for 5 randomly chosen products that we sell in Rohlik.
Your task is to explore these time series. Find what is the best price for which we can sell each product
in next 7 days (starting from the last day of time series). Goal is to maximize revenue (sales * sell_price)
while maintaining absolute margin - (sell_price - buy_price) * sales. For margin calculation, please count
that buy price for next 7 days will be same as it was in the last day of given time series. You can
enhance this data set with some external data, it’s up to you, be creative :-).

>● We prefer you to use python for this task.

>● Try to add comments to your code, so the reader knows what were your thoughts during this task.

>● Besides your python code, we would like to see clear output - what is proposed price, what are expected sales, what is expected revenue for each product.

>● Link to data - https://temp-roh.s3.eu-central-1.amazonaws.com/ml_task_data.zip"


## Results
The following table summarizes proposed prices for all products and estimated, sales, revenue and margin. All the values are rounded.

| id alias | id       | proposed price | price change | sales estimate | margin estimate | revenue estimate |
|----------|----------|----------------|--------------|----------------|-----------------|------------------|
| 0        | 82b9c... | 72             | 0            | 514            | 11190           | 36871            |
| 1        | 58fba... | 220            | 0            | 1482           | 97779           | 325918           |
| 2        | 42586... | 75             | -24          | 1545           | 8405            | 115367           |
| 3        | b2141... | 30             | -9           | 2406           | 6467            | 71127            |
| 4        | 56154... | 37             | -13          | 527            | 1041            | 19265            |



## The whole solution is divided into following scripts:
- `conf.py` - general configuration for all scripts
- `utils.oy` - functions shared among other scripts
- `eda.py` - exploratory data anlysis, geenrates figures which can be found in the `plots` folder. The product ids are mapped to numbers 0,1... for better readability, the mapping can be found in the `id_map.json` file.
- `sarimax.py` - modeling time series of sales using various models, but most importantly using SARIMAX (seasonal ARIMA with exogenous regressors). The model metrics can be found in `fit_eval_metrics.json` for fitted values and in `test_eval_metrics.json` for test predictions on last 14 days of data. Final models are stored as a pickle file `models.pickle` and list of exogenous columns/variables is stored in the `exog_cols.json` file.
- `revenue_optimisation.py` - solving the abovementioned revenue optimisation problem using model from previous step. The final proposed prices and according sales and revenue estimates with confidence intervals are stored in `optimal_scenarios.json`.

Code is structured with cell style used in spyder ide. You can run either interactively or generate results only.

## Notes
- Although some of the simpler models seemed to perform better on the test set, they are not able to directly incorporate the estimation of price influence on the sales and would require combination with other approach using price elasticity. For the sake of simplicity and clarity We decided to not go this way. This might require further investigation.
- The default seasonality is considered to be weekly since we are dealing with daily data and sales are expected to be influenced mainly by week cycle.
- No inputation was performed since the main used method can naturally deal with missing values and wrong imputation might distort the estimates.
- There is visible change in data after the start of the covid pandemic, however, the changes are not big enough to completely drop the preceding dates by default. Might be considered for experiment.


## Python and Packages
Python version used: 3.8.5
Required packages can be found in `requirements.txt`