ID_COL = "product_id"
DATE_COL = "date"
PRICE_COL = "sell_price"
MARGIN_COL = "margin"
SALES_COL = "sales"
BUY_PRICE_COL = "buy_price"

PROMOTION_CONF = {
    # at mostr one month of special price event duration
    "max_duration": 50,
    # 7 % change is considered to be noticeable a discount
    "price_change_threshold": .07,
    "price_col": PRICE_COL
    }

DATE_FORMAT = "%Y-%m-%d"

TRANSFORM_CONF = {
    "sort_cols": ["date"],
    "diff": {
        SALES_COL: [1],
        PRICE_COL: [1],
        },
    "pct_change": {
        SALES_COL: [1],
        PRICE_COL: [1],
        }
    }