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

The whole solution is divided to following scripts:
- `conf.py` - general configuration for all scripts
- `utils.oy` - functions shared among other scripts
- `eda.py` - exploratory data anlysis, geenrates figures which can be found in the `plots` folder, 
- `sarimax.py` - modeling time series of sales using various models, but most importantly using SARIMAX (seasonal ARIMA with exogenous regressors). The model metrics can be found in `fit_eval_metrics.json` for fitted values and in `test_eval_metrics.json` for test predictions on last 14 days of data. Final models are stored as a pickle file `models.pickle` and list of exogenous columns/variables is stored in the `exog_cols.json` file.
- `revenue_optimisation.py` - solving the abovementioned revenue optimisation problem using model from previous step. The final proposed prices and according sales and revenue estimates are stored for 