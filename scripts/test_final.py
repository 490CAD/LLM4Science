from typing import Optional, Union
import pandas as pd
import torch
import os
import logging
import torch.nn as nn
import tqdm
import numpy as np
from peft import peft_model
from datasets import Dataset
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

Super_Prompt = "Using background knowledge as a reference, choose one option as an answer from a multiple-choice question. The answer must begin with A or B or C or D or E.\nBackground knowledge is:BACK_KNOW\nThe question is:QUERY\nThe answer is:"

answer_to_label = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0}
label_to_label = {0:319, 1:350, 2:315, 3:360, 4:382}

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

logging.basicConfig(filename='log.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.DEBUG)

BASEDATA_PATH = "/home/kaggleLLAM/data/"
MODEL_PATH = "/home/kaggleLLAM/model/checkpoint-8332/"
# ADAPATER_PATH = "/home/kaggleLLAM/output/checkpoint-7520/"
MAX_SEQ_LEN = 10000000000
MAX_CON_LEN = 2500
res_list = []

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    padding_side="right",
    # pad_token="<|endoftext|>",
)

tokenizer.pad_token = tokenizer.unk_token

def preprocess(example):
    example['A'] = example['A'] if example['A'] is not None else "None"
    example['B'] = example['B'] if example['B'] is not None else "None"
    example['C'] = example['C'] if example['C'] is not None else "None"
    example['D'] = example['D'] if example['D'] is not None else "None"
    example['E'] = example['E'] if example['E'] is not None else "None"
    example = {k: v for k, v in example.items()}
    query = str(example['prompt']) + ' A: ' + str(example['A']) + ' B: ' + str(example['B']) + ' C: ' + str(example['C']) + ' D: ' + str(example['D']) + ' E: ' + str(example['E'])
    tokenized_example = tokenizer(Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(example['context'])[:1750])[:3500], return_tensors="pt", truncation=True)
    tokenized_example['input_ids'] = tokenized_example['input_ids'][0]
    tokenized_example['attention_mask'] = tokenized_example['attention_mask'][0]
    tokenized_example['label'] = answer_to_label[example['answer']]
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        ) # {"input_ids":tensor(BxS), "attention_mask":tensor(BxS), ...}
        batch = {k: v.view(batch_size, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels).view(batch_size, -1) # tensor(Bx1)
        return batch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        torch.cuda.empty_cache()
        labels = inputs.pop("labels", None)
        # forward pass
        outputs = model(**inputs).logits[:,-1,:]
        # index = torch.tensor([319, 350, 315, 360, 382]).to(f"cuda:{model.device.index}")
        # outputs = outputs.index_select(-1, index)
        # pro = torch.argsort(outputs, dim=-1, descending=True)[:,:3][0].detach().cpu().numpy()
        pro = torch.argsort(outputs, dim=-1, descending=True)[:,:3]
        # pro = torch.argmax(outputs, dim=-1,)
        answer_candicate = ""
        for answer in pro:
            token = tokenizer.decode(answer)
            answer_candicate += token
        res_list.append(answer_candicate)
        loss = torch.tensor(1.0, device=model.device.index)
        return (loss, outputs) if return_outputs else loss
    

def console_output(name, val):
    print('------------' + str(name) + '------------' )
    print(val)
    print('------------------------------------' )

def check_sub(res_list):
    import random
    ans = []
    for res in res_list:
        res_t = res.split(' ') # [AB, A]
        s = ""
        r = ['A', 'B', 'C', 'D', 'E'] # 
        cnt = 0
        for key in res_t:
            if cnt >= 3:
                break
            if key in ['A', 'B', 'C', 'D', 'E']:
                s = s + key # 'A'
                r.remove(key)
                cnt += 1
                if cnt != 3:
                    s = s + " "
            elif len(key) > 1 and key[0] in ['A', 'B', 'C', 'D', 'E']:
                s = s + key[0]
                r.remove(key[0])
                cnt += 1
                if cnt != 3:
                    s = s + " "
        while cnt < 3:
            random_index = random.randrange(len(r))
            s = s + r[random_index]
            r.pop(random_index)
            cnt += 1
            if cnt != 3:
                s = s + " "
        ans.append(s)
    return ans
    

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
    
    output_csv['prediction'] = check_sub(output_csv['prediction'])
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

if __name__ == "__main__":
    # cal_score(0, pd.read_csv(BASEDATA_PATH + '/n_eval.csv'), pd.read_csv( '/home/kaggleLLAM/data/submission2.csv'))
    # exit()
    # print(tokenizer.encode('A'), tokenizer.encode('B'), tokenizer.encode('C'), tokenizer.encode('D'), tokenizer.encode('E'))
    training_args = TrainingArguments(
        output_dir='/home/kaggleLLAM/output',
        per_device_eval_batch_size=1,
    )
    
    model_test = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
    
    # model_test = peft_model.PeftModelForCausalLM.from_pretrained(model_test, ADAPATER_PATH, adapter_name="SciQA_lora")
    
    trainer_test = CustomTrainer(
        model=model_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        args=training_args,
    )
    
#     # test
    test_data = pd.read_csv(BASEDATA_PATH + '/eval_wiki.csv')
    test_data['answer'] = 'A' # dummy answer that allows us to preprocess the test dataset just like we preprocessed the train dataset
    tokenized_test_dataset = Dataset.from_pandas(test_data).map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    sub = pd.DataFrame()
    sub['id'] = range(test_data.shape[0])
    sub['prediction'] = range(test_data.shape[0])
    # test_data = test_data.drop(columns='id')
    
    # Prediction_Output = trainer_test.predict(tokenized_test_dataset)
    # sub['prediction'] = res_list
    # sub.to_csv(BASEDATA_PATH+'submission.csv', index=False)
    preds = []
    pbar = tqdm(total=len(test_data))
    print('---------------------------------')
    model_test.eval()
    for _, row in tqdm(test_data.iterrows()):
        torch.cuda.empty_cache()
        query = str(row['prompt']) + ' A: ' + str(row['A']) + ' B: ' + str(row['B']) + ' C: ' + str(row['C']) + ' D: ' + str(row['D']) + ' E: ' + str(row['E'])
        prompt = Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(row['context'])[:MAX_CON_LEN])
        if len(prompt)>MAX_SEQ_LEN:
            answer_len = len(str(row['A']) + str(row['B']) + str(row['C']) + str(row['D']) + str(row['E']))
            reduce_nums_ratio = [int((0 - MAX_SEQ_LEN + len(prompt)) * len(str(row[word])) / answer_len) for word in 'ABCDE']
            query = str(row['prompt']) + ' A: ' + str(row['A'])[:-reduce_nums_ratio[0]] + ' B: ' + str(row['B'])[:-reduce_nums_ratio[1]] + ' C: ' + str(row['C'])[:-reduce_nums_ratio[2]] + ' D: ' + str(row['D'])[:-reduce_nums_ratio[3]] + ' E: ' + str(row['E'])[:-reduce_nums_ratio[4]]
            prompt = Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(row['context'])[:MAX_CON_LEN])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            first_token_probs = model_test(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:,-1,:][0]
        option_scores = first_token_probs[[319, 350, 315, 360, 382]].float().cpu().numpy() #ABCDE
        pred = np.array(["A", "B", "C", "D", "E"])[np.argsort(option_scores)[::-1][:3]]
        pred = ' '.join(pred)
        res_list.append(pred)
        pbar.update(1)
    pbar.close()
    sub['prediction'] = res_list
    sub.to_csv(BASEDATA_PATH+'submission2.csv', index=False)
    
    
    # cal_score(1, BASEDATA_PATH + '/new_eval.csv', sub)
    # cal_score(1, pd.read_csv(BASEDATA_PATH + '/new_eval.csv'), sub)
    


