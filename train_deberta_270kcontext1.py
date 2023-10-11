from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel
import torch.nn as nn
from map_at3_loss import map_at3_loss
import wandb
VER=103
# TRAIN WITH SUBSET OF 60K
NUM_TRAIN_SAMPLES = 1_024
# PARAMETER EFFICIENT FINE TUNING
# PEFT REQUIRES 1XP100 GPU NOT 2XT4
USE_PEFT = False
# NUMBER OF LAYERS TO FREEZE 
# DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_LAYERS = 18
# BOOLEAN TO FREEZE EMBEDDINGS
FREEZE_EMBEDDINGS = True
# LENGTH OF CONTEXT PLUS QUESTION ANSWER
MAX_INPUT = 256
# HUGGING FACE MODEL
MODEL = "/home/kaggleLLAM/model/deberta-v3-large-hf"


df_valid = pd.read_csv('/home/kaggleLLAM/data/train_with_context2.csv')
df_train = pd.read_csv('/home/kaggleLLAM/data/100k_context12_270kwiki_1002.csv')
# print(len(df_train))
df_train = df_train.fillna('').sample(len(df_train))
# df_train = df_train.fillna('').sample(1024) 

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(example):
    if example.get('context1'):
        first_sentence = [ "[CLS] " + example['context1'] ] * 5
    else:
        first_sentence = [ "[CLS] " + example['context'] ] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    # print(len(first_sentence), len(second_sentences))
    tokenized_example1 = tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example1['label'] = option_to_index[example['answer']]

    # first_sentence = [ "[CLS] " + example['context2'] ] * 5
    # second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    # # print(len(first_sentence), len(second_sentences))
    # tokenized_example2 = tokenizer(first_sentence, second_sentences, truncation='only_first', 
    #                               max_length=MAX_INPUT, add_special_tokens=False)
    # tokenized_example2['label'] = option_to_index[example['answer']]
    return tokenized_example1

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
        batch['labels'] = torch.tensor(labels, dtype=torch.float)
        return batch
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        torch.cuda.empty_cache()
        # inputs.to(f"cuda:{model.device.index}")
        labels = inputs.pop("labels", None)
        # forward pass
        outputs = model(**inputs).logits
        # outputs = model(**inputs).logits[:,-1,:]
        # pro = torch.argmax(outputs, dim=-1)
        # token =  tokenizer.decode(pro)
        # index = torch.tensor([319, 350, 315, 360, 382]).to(f"cuda:{model.device.index}")
        # score = outputs.index_select(1, index)
        labels = labels.long()
        Loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss = Loss_fn(input=outputs.view(len(labels),-1), target=labels.view(-1).long()).requires_grad_(True)

        return (loss, outputs) if return_outputs else loss

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}
if __name__ == "__main__":
    wandb.init(project='kaggleLLAM', name='deberta-v3-large-hf_bs4_lr2e-5_warm0.1_3epo_freeze12')
    dataset_valid = Dataset.from_pandas(df_valid)
    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.remove_columns(["__index_level_0__"])
    tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'context1', 'A', 'B', 'C', 'D', 'E', 'answer'])
    model = AutoModelForMultipleChoice.from_pretrained(MODEL)
    if FREEZE_EMBEDDINGS:
        print('Freezing embeddings.')
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
    if FREEZE_LAYERS>0:
        print(f'Freezing {FREEZE_LAYERS} layers.')
        for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
            for param in layer.parameters():
                param.requires_grad = False
    training_args = TrainingArguments(
        warmup_ratio=0.1, 
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        report_to='wandb',
        output_dir = f'./checkpoints_{VER}',
        overwrite_output_dir=True,
        fp16=False,
        gradient_accumulation_steps=8,
        logging_steps=25,
        evaluation_strategy='steps',
        eval_steps=1000,
        save_strategy="epoch",
        save_steps=1000,
        load_best_model_at_end=False,
        metric_for_best_model='map@3',
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics = compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(f'model_v{VER}')