DATE_FORMAT = "%Y-%m-%d"

ID_COL = "product_id"
TRANSFORM_CONF = {
    "sort_cols": ["date"],
    "diff": {
        "sales": [1],
        "sell_price": [1],
        },
    "pct_change": {
        "sales": [1],
        "sell_price": [1],
        }
    }