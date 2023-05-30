ID_COL = "product_id"
DATE_COL = "date"
PRICE_COL = "sell_price"
MARGIN_COL = "margin"
SALES_COL = "sales"

PROMOTION_CONF = {
    "max_duration": 7,
    "price_drop_threshold": -.1
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