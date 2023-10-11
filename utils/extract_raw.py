import os
import pandas as pd

path = "/home/kaggleLLAM/data"

file_path = path + "/all_12_with_context2.csv"

df = pd.read_csv(file_path)

new_order = ['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']

df = df[new_order]

df.to_csv(path +"/all_12.csv", index=False)