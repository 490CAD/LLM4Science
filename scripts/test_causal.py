from typing import Optional, Union
import pandas as pd
import torch
import os
import logging
import torch.nn as nn
from peft import peft_model
from ast import literal_eval
from datasets import Dataset, load_metric
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

Super_Prompt = "Choose one of the following multiple-choice questions as your answer response, which must begin with A or B or C or D or E.\nquery\nThe answer is:"

answer_to_label = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0}

logging.basicConfig(filename='log.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.DEBUG)

BASEDATA_PATH = "/home/kaggleLLAM/data/"
MODEL_PATH = "/home/kaggleLLAM/model/Nous-Hermes-llama-2-7b/"
ADAPATER_PATH = "/home/kaggleLLAM/output/checkpoint-4252/"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    padding_side="left",
    # pad_token="<|endoftext|>",
)

tokenizer.pad_token = tokenizer.unk_token

def preprocess(example):
    print(example['prompt'])
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
        batch['labels'] = torch.tensor(labels).view(batch_size, -1) # tensor(Bx1)
        return batch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        # forward pass
        outputs = model(**inputs).logits[:,-1,:]
        # index = torch.tensor([319, 350, 315, 360, 382]).to(f"cuda:{model.device.index}")
        # outputs = outputs.index_select(-1, index)
        print(outputs)
        pro = torch.argsort(outputs, dim=-1, descending=True)[:,:3]
        # pro = torch.argmax(outputs, dim=-1,)
        token_list = []
        for answer in pro:
            token = tokenizer.decode(answer)
            token_list.append(token)
        print(token_list)
        exit()
        score = outputs.index_select(1, index)
        Loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss = Loss_fn(input=score.view(len(labels),-1), target=labels.view(-1).long()).requires_grad_(True)
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    training_args = TrainingArguments(
        output_dir='/home/kaggleLLAM/output',
        per_device_eval_batch_size=20,
    )
    
    model_test = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
    
    model_test = peft_model.PeftModelForCausalLM.from_pretrained(model_test, ADAPATER_PATH, adapter_name="SciQA_lora")
    
    trainer_test = CustomTrainer(
        model=model_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        args=training_args,
    )
    
#     # test
    test_data = pd.read_csv(BASEDATA_PATH + '/new_eval.csv')[26:27]
    # test_df['answer'] = 'A' # dummy answer that allows us to preprocess the test dataset just like we preprocessed the train dataset
    tokenized_test_dataset = Dataset.from_pandas(test_data).map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    Prediction_Output = trainer_test.predict(tokenized_test_dataset)
    # print(torch.tensor(Prediction_Output.predictions), Prediction_Output.label_ids, Prediction_Output.metrics)