import pandas as pd

DATA_PATH = "/home/kaggleLLAM/data/dataset_wiki_new_1_balanced.csv"
df = pd.read_csv(DATA_PATH)
df['id'] = [i for i in range(300)]
new_order = ['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']
df = df[new_order]
df.to_csv('/home/kaggleLLAM/data/combined.csv', index=False)