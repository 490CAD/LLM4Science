import pandas as pd
import random

DATA_PATH = "/home/kaggleLLAM/data/check/valid_mmlu_1526_ind0.csv"
df = pd.read_csv(DATA_PATH)
df['E'] = ['' for i in range(len(df))]
new_order = ['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']
df = df[new_order]
for index, line in df.iterrows():
    answer = line['answer']
    idx = ['A', 'B', 'C', 'D']
    idx.remove(answer)
    key = random.randint(0, 2)
    # line['E'] = line[idx[key]]
    # print(idx, key)
    df['E'][index] = line[idx[key]]
    # break
df.to_csv('/home/kaggleLLAM/data/check/fixed.csv', index=False)