import pandas as pd

def map_ids(df: pd.DataFrame, id_col: str):
    """
    Maps values of id_col to string sequence '0', '1', ...
    Serves only for better readability.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Column on which to perform mapping.

    Returns
    -------
    Transformed df and mapping.

    """
    
    old_ids = set(df[id_col])
    ids = [str(i) for i in range(len(old_ids))]
    id_map = dict(zip(old_ids, ids))
    
    df[id_col] = df[id_col].map(id_map)
    
    return df, id_map


def split_df(df: pd.DataFrame, id_col: str) -> dict:
    """
    Splits dataframe to several smaller dataframes by a selected id column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be splited to several smaller dataframes.
    id_col : str
        Column to split by.

    Returns
    -------
    dict
        Dict of id and corresponding dataframe.

    """        
    pids = sorted(list(set(df[id_col])))
    pid_df = {pid : df[df[id_col] == pid] for pid in pids}

    return pid_df   


def init_transform(
        df: pd.DataFrame,
        date_col: str,
        date_format: str,
        price_col: str,
        margin_col: str,
        id_col: str
        ):
    """
    Initital transformation of our data. 
    Trasforming `date` column to specified format.
    Adding `buy price` column as a difference of the sell_price and margin.
    Mapping ids to more readable values.

    Parameters
    ----------
    df : pd.DataFrame
        DF to be transformed.
    date_col : str
        Date column name.
    date_format : str
        Format of date column.
    price_col : str
        Price column name.
    margin_col : str
        Margin column name.
    id_col : str
        ID column name.

    Returns
    -------
    df_prices : TYPE
        Transformed df.
    id_map : TYPE
        DESCRIPTION.

    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    
    if price_col in df.columns and margin_col in df.columns:
        df["buy_price"] = df[price_col] - df[margin_col]

    # map ids fro better readability
    id_map = None
    if id_col in df.columns:
        df, id_map = map_ids(df, id_col)

    return df, id_map


def general_transform(
        df: pd.DataFrame, transform_conf: dict
    ) -> pd.DataFrame:
    """
    Transforms a datframe of sales and prices data.
    Currently only sorting and pct_change is implemented
    
    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    transform_conf : dict
        Config of individual transformations to be performed.
        Sample input: {"pct_change": {"sell_price": [1]}}

    Returns
    -------
    pd.DataFrame
        Transformed dataframe.

    """
    if sort_cols := transform_conf.get("sort_cols"):
        df = df.sort_values(sort_cols)
        
    # differentiating
    col_diff_td = transform_conf.get("diff", {})
    # columns to lists of time lags
    for dc, tdeltas in col_diff_td.items():
        if dc in df.columns:
            for td in tdeltas:
                df[f"{dc}_diff_{td}"] = df[dc].diff(td)
    
    # `pct_change` transformation
    col_pc_td = transform_conf.get("pct_change", {})
    # columns to lists of time lags
    for pcc, tdeltas in col_pc_td.items():
        if pcc in df.columns:
            for td in tdeltas:
                df[f"{pcc}_pct_change_{td}"] = df[pcc].pct_change(td)
                
    return df
