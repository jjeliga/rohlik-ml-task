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


def transform_vars(
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
    
    # `pct_change` transformation
    col_pc_td = transform_conf.get("pct_change", {})
    # columns to lists of time lags
    for pcc, tdeltas in col_pc_td.items():
        if pcc in df.columns:
            for td in tdeltas:
                df[f"{pcc}_pct_change_{td}"] = df[pcc].pct_change(td)
                
    return df
