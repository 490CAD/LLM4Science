import os
import pandas as pd

A_csv = pd.read_csv("/home/kaggleLLAM/data/train.csv").drop("id", axis=1)
B_csv = pd.read_csv("/home/kaggleLLAM/data/dataset_wiki_new_1_balanced.csv")

combine_csv = pd.concat([A_csv, B_csv], ignore_index=True)
combine_csv.to_csv("/home/kaggleLLAM/data/test_fixed.csv", index=False)
# combine_csv = pd.read_csv("/home/kaggleLLAM/data/all_100k.csv")
# for index, row in combine_csv.iterrows():
#     if index == 29:
#         print(row['prompt'])
#         print(row['A'])
#         print(row['B'])
#         print(row['C'])
#         print(row['D'])
#         print(row['E'])
#         print(row['answer'])
#         print(index)
#         exit()
print(len(combine_csv))
print(combine_csv.head)