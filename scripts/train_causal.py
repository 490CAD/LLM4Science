from typing import Optional, Union
import pandas as pd
import wandb
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import Dataset, load_metric
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, top_k_top_p_filtering

Super_Prompt = "Choose one of the following multiple-choice questions as your answer response, which must begin with A or B or C or D or E.\nquery\nThe answer is:"

MODEL_PATH = "/home/kaggleLLAM/model/8f95aa9cd207db7b24179fc779c2b8973e71bee2"
BASEDATA_PATH = "/home/kaggleLLAM/data"
answer_to_label = {'A':319, 'B':350, 'C':315, 'D':360, 'E':382} # [319, 350, 315, 360, 382]
# answer_to_label = {'A':319, 'B':350, 'C':315, 'D':360, 'E':382}
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    padding_side="right",
    pad_token="<unk>",
)

# tokenizer.pad_token = tokenizer.unk_token

def console_output(name, val):
    print('------------' + str(name) + '------------' )
    print(val)
    print('------------------------------------' )

def preprocess(example):
    example['A'] = example['A'] if example['A'] is not None else "None"
    example['B'] = example['B'] if example['B'] is not None else "None"
    example['C'] = example['C'] if example['C'] is not None else "None"
    example['D'] = example['D'] if example['D'] is not None else "None"
    example['E'] = example['E'] if example['E'] is not None else "None"
    example = {k: v for k, v in example.items()}
    query = example['prompt'] + ' A: ' + example['A'] + ' B: ' + example['B'] + ' C: ' + example['C'] + ' D: ' + example['D'] + ' E: ' + example['E']
    tokenized_example = tokenizer(Super_Prompt.replace('query', query), return_tensors="pt", truncation=True)
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
        batch['labels'] = torch.tensor(labels).view(batch_size, -1) # tensor(Bx2) or tensor(Bx1)
        return batch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        torch.cuda.empty_cache()
        labels = inputs.pop("labels", None)
        # forward pass
        outputs = model(**inputs).logits[:,-1,:]
        # pro = torch.argmax(outputs, dim=-1)
        # token =  tokenizer.decode(pro)
        # index = torch.tensor([319, 350, 315, 360, 382]).to(f"cuda:{model.device.index}")
        # score = outputs.index_select(1, index)
        Loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss = Loss_fn(input=outputs.view(len(labels),-1), target=labels.view(-1).long()).requires_grad_(True)
        return (loss, outputs) if return_outputs else loss


if __name__  == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    #wandb_init
    wandb.init(project='kaggleLLAM', name='llama2-13b_cau_bs2_lr5e-5_warm0.2_lora_r16_1epo')
    
    # prepare the dataset
    train_data = pd.read_csv(BASEDATA_PATH + '/new_train.csv')
    train_data = train_data.drop(columns="id")
    console_output("train_data_shape", train_data.shape)
    eval_data = pd.read_csv(BASEDATA_PATH + '/new_eval.csv')
    eval_data = eval_data.drop(columns="id")
    console_output("eval_data_shape", eval_data.shape)
    
    # init dataset & models
    train_dataset = Dataset.from_pandas(train_data)
    console_output('train_dataset', train_dataset)
    eval_dataset = Dataset.from_pandas(eval_data)
    console_output('eval_dataset', eval_dataset)
    tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    console_output("tokenized_train_dataset", tokenized_train_dataset)
    tokenized_eval_dataset = eval_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    console_output("tokenized_eval_dataset", tokenized_eval_dataset)
    
    
    # train
    training_args = TrainingArguments(
        warmup_ratio=0.2,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        report_to='wandb',
        output_dir='/home/kaggleLLAM/output',
        save_total_limit=1,
        save_strategy='epoch',
        evaluation_strategy='steps',
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=10,
        # deepspeed="/home/kaggleLLAM/utils/stage2.json"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
    
    peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )
    trainer.train()