WORKDIR = ""
DATE_FORMAT = "%Y-%m-%d"

ID_COL = "product_id"
TRANSFORM_CONF = {
    "sort_cols": ["date"],
    "pct_change": {"sell_price": [1]}
    }