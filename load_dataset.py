# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:58:53 2022

@author: devenv
"""

import pandas as pd
import numpy as np

#loads the dataset
def load_dataset(dataset_path   = 'dataset/',               #folder where the dataset is placed
                 dataset_tr_set = 'abstracts_train.csv',    #instances file
                 dataset_tr_lab = 'entities_train.csv'):    #targets file

    #reads the dataset
    X_raw = pd.read_csv(dataset_path + dataset_tr_set, sep='\t')
    y_raw = pd.read_csv(dataset_path + dataset_tr_lab, sep='\t', index_col = 'id')
    
    #merges data and title together.
    X_raw["data"] = X_raw["title"] + " " + X_raw["abstract"]
    
    #removes abstract and title since we don't need it anymore
    X_raw.drop(["abstract"], axis=1, inplace=True)
    X_raw.drop(["title"], axis=1, inplace=True)
    
    #safety
    X = X_raw.copy()
    
    #creates the targets' dataframe
    y = pd.DataFrame()
    y['abstract_id'] = y_raw['abstract_id'].copy().drop_duplicates()
    y['data'] = [list() for x in range(len(y.index))]
    
    #stores in the column 'data' an array containing the various entities
    for i in range(len(X)):
        tags = y_raw.loc[y_raw['abstract_id'] == X.iloc[i]["abstract_id"]]
        
        for j, row in tags.iterrows():
            offset_start = row['offset_start']
            offset_finish = row['offset_finish']
            entity_type = row['type']
            mention = row['mention']
            entity_ids = row['entity_ids']
            
            to_insert = (offset_start, offset_finish, mention, entity_type, entity_ids)
            
            #working
            y.iloc[i]['data'].append(to_insert)
    
    #index reset
    X.reset_index(inplace = True, drop = True)
    y.reset_index(inplace = True, drop = True)
    
    return X, y