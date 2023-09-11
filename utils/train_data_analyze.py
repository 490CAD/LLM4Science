import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = pd.read_csv('/home/kaggleLLAM/data/eval_context_66855.csv')
Super_Prompt = "Using background knowledge as a reference, choose one option as an answer from a multiple-choice question. The answer must begin with A or B or C or D or E.\nBackground knowledge is:BACK_KNOW\nThe question is:QUERY\nThe answer is:"

d = {}

for idx, row in file.iterrows():
    query = str(row['prompt']) + ' A: ' + str(row['A']) + ' B: ' + str(row['B']) + ' C: ' + str(row['C']) + ' D: ' + str(row['D']) + ' E: ' + str(row['E'])
    # context = Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(row['context']))[:30000]
    prompt = Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(row['context'])[:1750])
    if d.get(len(prompt)) is None:
        d[len(prompt)] = 1
    else:
        d[len(prompt)] += 1

d = sorted(d.items(), key=lambda s:s[0])
for item in d:
    plt.bar(item[0], item[1])
plt.savefig('/home/kaggleLLAM/maxLength.png')

print(d)