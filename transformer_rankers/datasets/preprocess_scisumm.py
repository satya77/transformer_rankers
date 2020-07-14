from IPython import embed
from tqdm import tqdm
from os import walk,listdir
from os.path import join
from transformer_rankers.datasets.scicummGen import Paper,Annotation

import csv
import gzip
import codecs

import pandas as pd
import numpy as np

def transform_to_dfs(path):
    """
    Transforms TREC 2020 Passage Ranking files (https://microsoft.github.io/TREC-2020-Deep-Learning/)
    to train, valid and test dfs containing only positive query-passage combinations.

    Args:
        path: str with the path for the TREC folder containing: 
            - collection.tar.gz (uncompressed: collection.tsv)
            - queries.tar.gz (uncompressed: queries.train.tsv, queries.dev.tsv)
            - qrels.dev.tsv
            - qrels.train.tsv
        
    Returns: (train, valid, test) pandas DataFrames
    """
    train=[]
    for directory in listdir(path):
        path_dir=join(path, directory)
        annotations_path = join(path_dir, 'annotation')
        annotation = Annotation(join(annotations_path, listdir(annotations_path)[0]))
        # For all the citances
        for citance in annotation.citances:
            train.append([citance["query"], citance["passage"]])
    train_df_all = pd.DataFrame(train, columns=["query", "passage"])

    msk = np.random.rand(len(train_df_all)) < 0.8
    train_df = train_df_all[msk]
    dev_df = train_df_all[~msk]
    return train_df, dev_df