import numpy as np
import os
import pandas as pd
from pandas import DataFrame as df

file_path_train = "/home/kaggleLLAM/data/train_66855_0904.csv"
# file_path_eval = "/home/kaggleLLAM/data/eval.csv"

file_train = pd.read_csv(file_path_train)

# file_eval =  pd.read_csv(file_path_eval)

# merge_file = pd.concat([file_train, file_eval], ignore_index=True)
merge_file = file_train
shuffle_file = merge_file.sample(frac=1).reset_index(drop=True)

split_point = len(shuffle_file) // 10 * 9

shuffle_train = shuffle_file[:split_point]
shuffle_train = shuffle_train.copy()
shuffle_train['id'] = range(len(shuffle_train))
shuffle_eval = shuffle_file[split_point:]
shuffle_eval = shuffle_eval.copy()
shuffle_eval['id'] = range(len(shuffle_eval))

shuffle_train.to_csv("/home/kaggleLLAM/data/train_66855.csv", index=False)
shuffle_eval.to_csv("/home/kaggleLLAM/data/eval_66855.csv", index=False)

