import pandas as pd
import numpy as np

def transform_to_dfs(path):
    """
   does the train and test split for the already ranked data

    Args:
        path: str with the csv file with the ranked files

        
    Returns: (train, valid, test) pandas DataFrames
    """
    np.random.seed(42)

    train_df_all = pd.read_csv(path , sep="\t", header=None)
    train_df_all.columns=['query','passage','label','id']

    msk = np.random.rand(len(train_df_all['query'].unique())) < 0.85

    train_queries = train_df_all['query'].unique()[msk]
    dev_queries =  train_df_all['query'].unique()[~msk]

    train_df = train_df_all[train_df_all['query'].isin(list(train_queries))]
    dev_df = train_df_all[train_df_all['query'].isin(list(dev_queries))]

    return train_df, dev_df

