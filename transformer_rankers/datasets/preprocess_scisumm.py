from os import listdir
from os.path import join
from transformer_rankers.datasets.scicummGen import Paper,Annotation

import csv
import gzip
import codecs

import pandas as pd
import numpy as np

def transform_to_dfs(path):
    """
    Transforms the scisumm dataset to pandas datafram and do random development split
        
    Returns: (train, valid) pandas DataFrames
    """
    train=[]
    for directory in listdir(path):
        if directory !="scisumm":
            path_dir=join(path, directory)
            annotations_path = join(path_dir, 'annotation')
            annotation = Annotation(join(annotations_path, listdir(annotations_path)[0]))
            # For all the citances
            for citance in annotation.citances:
                train.append([citance["query"], citance["passage"]])
    train_df_all = pd.DataFrame(train, columns=["query", "passage"])

    msk = np.random.rand(len(train_df_all)) < 0.85
    train_df = train_df_all[msk]
    dev_df = train_df_all[~msk]
    return train_df, dev_df