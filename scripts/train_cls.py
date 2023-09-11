from typing import Optional, Union
import pandas as pd
import wandb
import torch
import torchvision
import logging
import os
import torch.nn as nn
from ast import literal_eval
from datasets import Dataset, load_metric
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification

logging.basicConfig(filename='log.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.DEBUG)

MODEL_PATH = "/home/kaggleLLAM/model/llama2-13b-8bit"
BASEDATA_PATH = "/home/kaggleLLAM/data"
option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    padding_side="left",
    pad_token="<|endoftext|>",
)
train_data = pd.DataFrame(data=None, columns=['senquence', 'label'])
eval_data = pd.DataFrame(data=None, columns=['senquence', 'label'])
test_data = pd.DataFrame(data=None, columns=['senquence', 'label'])

def console_output(name, val):
    print('------------' + str(name) + '------------' )
    print(val)
    print('------------------------------------' )

def data_generate(csvfile, generate_data, filename):
    for _, row in csvfile.iterrows():
        for choice in 'ABCDE':
            senquence = "<|prompt|>The question is:"+str(row['prompt'])+"</s><|answer|>The correct answer is:"+str(row[choice])
            label = 1.0 if row['answer']==choice else 0.0
            generate_data.loc[len(generate_data)] = {"senquence":senquence, "label":label}
    generate_data.to_csv("/home/kaggleLLAM/data/"+filename)
    return generate_data
    
def preprocess(example):
    example = {k: v for k, v in example.items()}
    tokenized_example = tokenizer(example['senquence'], truncation=True)
    tokenized_example['label'] = literal_eval(example['label']) if type(example['label'])==str else example['label']
    return tokenized_example

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logging.debug(str(nn.Softmax(dim=-1)(logits))+"\n"+str(labels))
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = torchvision.ops.sigmoid_focal_loss(logits, labels, reduction="mean", alpha=-1)
        logging.debug(str(loss))
        # ce_loss = nn.functional.cross_entropy(input=logits.view(-1, self.model.config.num_labels), target=labels.view(-1), reduction="none") 
        return (loss, outputs) if return_outputs else loss

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
        batch['labels'] = torch.tensor(labels).view(batch_size, -1) # tensor(Bx2) or tensor(Bx1)
        return batch


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    #wandb_init
    wandb.init(project='kaggleLLAM', name='llama2-13b-8bit_bs2*8_lr5e-5_warm0.2_focal_10epo')
    
    # prepare the dataset
    train_data = pd.read_csv(BASEDATA_PATH + '/train_data.csv')[:20]
    train_data = train_data.drop(columns="id")
    console_output("train_data_shape", train_data.shape)
    eval_data = pd.read_csv(BASEDATA_PATH + '/train_data.csv')[20:30]
    eval_data = eval_data.drop(columns="id")
    console_output("eval_data_shape", eval_data.shape)
    # df_train, df_eval = pd.read_csv(BASEDATA_PATH + '/train.csv'), pd.read_csv(BASEDATA_PATH + '/eval.csv')
    # train_data = data_generate(df_train, train_data, "train_data_2.csv")
    # eval_data = data_generate(df_eval, eval_data, "eval_data.csv")
    # exit()
    
    # init dataset & models
    train_dataset = Dataset.from_pandas(train_data)
    console_output('train_dataset', train_dataset)
    eval_dataset = Dataset.from_pandas(eval_data)
    console_output('eval_dataset', eval_dataset)
    tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=['senquence', 'label'])
    console_output("tokenized_train_dataset", tokenized_train_dataset)
    tokenized_eval_dataset = eval_dataset.map(preprocess, remove_columns=['senquence', 'label'])
    console_output("tokenized_eval_dataset", tokenized_eval_dataset)
    
    
    # train
    training_args = TrainingArguments(
        warmup_ratio=0.2,
        learning_rate=5e-5,
        per_device_train_batch_size=20,
        # gradient_accumulation_steps=2,
        per_device_eval_batch_size=10,
        num_train_epochs=100,
        report_to='none',
        output_dir='/home/kaggleLLAM/output',
        save_total_limit=3,
        save_strategy='no',
        evaluation_strategy='steps',
        eval_steps=5,
        logging_strategy="steps",
        logging_steps=1,
        # deepspeed="/home/kaggleLLAM/utils/stage2.json"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        num_labels=2,
)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    trainer.train()
