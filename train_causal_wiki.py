from typing import Optional, Union
import pandas as pd
import wandb
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig, AdaLoraModel
from datasets import Dataset, load_metric
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, top_k_top_p_filtering

Super_Prompt = "Using background knowledge as a reference, choose one option as an answer from a multiple-choice question. The answer must begin with A or B or C or D or E.\nBackground knowledge is:BACK_KNOW\nThe question is:QUERY\nThe answer is:"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
MODEL_PATH = "/home/kaggleLLAM/model/llama2-7b"
BASEDATA_PATH = "/home/kaggleLLAM/data"
MAX_SEQ_LEN = 5601 - 1750 + 3000
MAX_PRO_LEN = 3000
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
    query = str(example['prompt']) + ' A: ' + str(example['A']) + ' B: ' + str(example['B']) + ' C: ' + str(example['C']) + ' D: ' + str(example['D']) + ' E: ' + str(example['E'])
    prompt = Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(example['context'])[:MAX_PRO_LEN])
    if len(prompt)>MAX_SEQ_LEN:
        answer_len = len(str(example['A']) + str(example['B']) + str(example['C']) + str(example['D']) + str(example['E']))
        reduce_nums_ratio = [int((len(prompt) - MAX_SEQ_LEN) * len(str(example[word])) / answer_len) for word in 'ABCDE']
        query = str(example['prompt']) + ' A: ' + str(example['A'])[:-reduce_nums_ratio[0]] + ' B: ' + str(example['B'])[:-reduce_nums_ratio[1]] + ' C: ' + str(example['C'])[:-reduce_nums_ratio[2]] + ' D: ' + str(example['D'])[:-reduce_nums_ratio[3]] + ' E: ' + str(example['E'])[:-reduce_nums_ratio[4]]
        prompt = Super_Prompt.replace('QUERY', query).replace('BACK_KNOW', str(example['context'])[:MAX_PRO_LEN])
        tokenized_example = tokenizer(prompt, return_tensors="pt", truncation=True)
    else:
        tokenized_example = tokenizer(prompt, return_tensors="pt", truncation=True)
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
        # inputs.to(f"cuda:{model.device.index}")
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
    
    #wandb_init
    wandb.init(project='kaggleLLAM', name='llama2-7b_cau_bs1_lr5e-5_warm0.2_lora_r16_2epo_wiki66855')
    
    # prepare the dataset
    train_data = pd.read_csv(BASEDATA_PATH + '/train_context_66855.csv')
    # train_data = train_data.drop(columns="id")
    console_output("train_data_shape", train_data.shape)
    eval_data = pd.read_csv(BASEDATA_PATH + '/eval_context_66855.csv')
    # eval_data = eval_data.drop(columns="id")
    console_output("eval_data_shape", eval_data.shape)
    
    # init dataset & models
    train_dataset = Dataset.from_pandas(train_data)
    console_output('train_dataset', train_dataset)
    eval_dataset = Dataset.from_pandas(eval_data)
    console_output('eval_dataset', eval_dataset)
    tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    console_output("tokenized_train_dataset", tokenized_train_dataset)
    tokenized_eval_dataset = eval_dataset.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    console_output("tokenized_eval_dataset", tokenized_eval_dataset)
    
    
    # train
    training_args = TrainingArguments(
        warmup_ratio=0.2,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        report_to='wandb',
        output_dir='/home/kaggleLLAM/output',
        save_total_limit=2,
        save_strategy='epoch',
        evaluation_strategy='steps',
        eval_steps=1500,
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
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
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
