import pandas as pd
import numpy as np

def transform_to_dfs(path,test_path=None,dev_path=None):
    """
   does the train and test split for the already ranked data

    Args:
        path: str with the csv file with the ranked files

        
    Returns: (train, valid, test) pandas DataFrames
    """
    np.random.seed(42)

    train_df_all = pd.read_csv(path , sep="\t", header=None)
    train_df_all.columns=['query','passage','label','id']
    if test_path!=None:
        test_df= pd.read_csv(test_path , sep="\t", header=None)
        test_df.columns=['query','passage','label','id']
    else:
        test_df=None
    if dev_path==None:
        msk = np.random.rand(len(train_df_all['query'].unique())) < 0.9

        train_queries = train_df_all['query'].unique()[msk]
        dev_queries =  train_df_all['query'].unique()[~msk]

        train_df = train_df_all[train_df_all['query'].isin(list(train_queries))]
        dev_df = train_df_all[train_df_all['query'].isin(list(dev_queries))]
    else:
        train_df = train_df_all
        dev_df= pd.read_csv(dev_path , sep="\t", header=None)
        dev_df.columns = ['query','passage','label','id']
    return train_df, dev_df , test_df

