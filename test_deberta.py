import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bfs

from collections.abc import Iterable

import faiss
from faiss import write_index, read_index

from sentence_transformers import SentenceTransformer

import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

from dataclasses import dataclass
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims = True)    

test_df = pd.read_csv("/home/kaggleLLAM/data/train_with_context2.csv")
test_df.index = list(range(len(test_df)))
test_df['id'] = list(range(len(test_df)))
test_df["prompt"] = test_df["context"].apply(lambda x: x[:2500]) + " #### " +  test_df["prompt"]

model_dir = "/home/kaggleLLAM/model/deberta-v3-large-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# We'll create a dictionary to convert option names (A, B, C, D, E) into indices and back again
options = 'ABCDE'
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}


# https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U

def console_output(name, val):
    print('------------' + str(name) + '------------' )
    print(val)
    print('------------------------------------' )

def preprocess(example):
    # The AutoModelForMultipleChoice class expects a set of question/answer pairs
    # so we'll copy our question 5 times before tokenizing
    first_sentence = [example['prompt']] * 5
    second_sentence = []
    for option in options:
        second_sentence.append(example[option])
    # Our tokenizer will turn our text into token IDs BERT can understand
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

tokenized_test_dataset = Dataset.from_pandas(test_df[['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']].drop(columns=['id'])).map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["__index_level_0__"])
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

model_ckpts = {
    "/home/kaggleLLAM/model_v2": 1,
    "/home/kaggleLLAM/model_0914": 1,
    '/home/kaggleLLAM/used_models/2023kagglellm-deberta-v3-large-model1': 1,
    '/home/kaggleLLAM/used_models/llm-science-run-context-2': 1,
    '/home/kaggleLLAM/used_models/llm-se-debertav3-large': 1,
    '/home/kaggleLLAM/used_models/my-1-epoch': 1,
    '/home/kaggleLLAM/used_models/science-exam-trained-model-weights/run_0':1,
    '/home/kaggleLLAM/used_models/science-exam-trained-model-weights/run_1':1,
    '/home/kaggleLLAM/used_models/science-exam-trained-model-weights/run_2':1,
}

ans_df =  pd.read_csv('/home/kaggleLLAM/data/n_eval.csv')

preds = []
for ckpt in tqdm(model_ckpts.keys()):
    print(ckpt + ':' + str(model_ckpts[ckpt]))

    model = AutoModelForMultipleChoice.from_pretrained(ckpt).cuda()
    model.eval()
    
    test_predictions = []
    for batch in tqdm(test_dataloader):
        for k in batch.keys():
            batch[k] = batch[k].cuda()
        with torch.no_grad():
            outputs = model(**batch)
        test_predictions.append(outputs.logits.cpu().detach())
    predictions = torch.cat(test_predictions)
    print(predictions.shape)
    preds.append(softmax((predictions * model_ckpts[ckpt]).numpy()))

    del model
pro_list = [1 for i in range(len(preds))]
b_m = 0
tmp_pred = preds[0]
for i in range(len(preds) - 1):
    pro = 0
    for k in range(100):
        tp = (i / 100) * tmp_pred + (1 - i / 100) * preds[i + 1]

        predictions_as_ids = np.argsort(-tp, 1)

        predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
        
        predictions_as_string = test_df['prediction'] = [
            ' '.join(row) for row in predictions_as_answer_letters[:, :3]
        ]

        m = MAP_at_3(predictions_as_string, ans_df.answer.values)
        if m > b_m:
            b_m = m
            pro = i
    if pro == 0:
        pro_list[i + 1] = 0
    else:
        tmp_pred = pro / 100 * tmp_pred + (1 - pro / 100) * preds[i + 1]
        for j in range(0, i + 1):
            pro_list[j] *= pro / 100
        pro_list[i + 1] *= (100 - pro) / 100

console_output("best score", b_m)
console_output("best pro_list", pro_list)


# predictions_as_ids = np.argsort(-preds, 1)

# predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
# # predictions_as_answer_letters[:3]

# predictions_as_string = test_df['prediction'] = [
#     ' '.join(row) for row in predictions_as_answer_letters[:, :3]
# ]

submission = test_df[['id', 'prediction']]
submission.to_csv('submission.csv', index=False)



def cal_score(mode, test_csv, output_csv):
    """
        mode means input style
            if mode == 0 means test_csv & output csv are pandas dataforme
            if mode == 1 means test_csv is file path but output_csv is pandas dataforme
        test_csv means the original test data
        output_csv means the submission data
        test_csv's head is like "id, prompt, A, B, C, D, E, answer"
        output_csv's head is like "id, preceission"
    """
    if mode in [1, 2]:
        test_df = pd.read_csv(test_csv)
        test_csv = test_df
        if mode == 2:
            output_df = pd.read_csv(output_csv)
            output_csv = output_df

    ans_dict = {}
    for index, line in test_csv.iterrows():
        id_value, answer_value = line['id'], line['answer']
        ans_dict[id_value] = answer_value
    
    # output_csv['prediction'] = check_sub(output_csv['prediction'])
    cnt = 0
    sum = 0.0
    for index, line in output_csv.iterrows():
        id_value, predict_value = line['id'], line['prediction'].split(' ')
        cnt += 1
        temp = 0
        for ans in predict_value:
            temp += 1
            if (len(ans)>1 and ans[0] in ['A','B','C','D','E']) or len(ans)==1:
                if (len(ans)>1 and ans[0]==ans_dict[id_value]) or (len(ans)==1 and ans==ans_dict[id_value]):
                    sum += 1.0 / temp
                    break
            else:
                continue
            
    console_output("CV Score", sum / cnt)

# cal_score(0, pd.read_csv('/home/kaggleLLAM/data/n_eval.csv'), pd.read_csv( '/home/kaggleLLAM/submission.csv'))
    