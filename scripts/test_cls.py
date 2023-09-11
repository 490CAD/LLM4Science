from typing import Optional, Union
import pandas as pd
import torch
import os
import logging
import torch.nn as nn
import torchvision
from ast import literal_eval
from datasets import Dataset, load_metric
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification

logging.basicConfig(filename='log.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.DEBUG)

BASEDATA_PATH = "/home/kaggleLLAM/data/"
MODEL_PATH = "/home/kaggleLLAM/model/llama2-13b-8bit"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    padding_side="left",
    pad_token="<|endoftext|>",
)

def preprocess(example):
    example = {k: v for k, v in example.items()}
    tokenized_example = tokenizer(example['senquence'], truncation=True)
    tokenized_example['label'] = literal_eval(example['label']) if type(example['label'])==str else example['label']
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
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = torchvision.ops.sigmoid_focal_loss(logits, labels, reduction="mean", alpha=-1)
        # ce_loss = nn.functional.cross_entropy(input=logits.view(-1, self.model.config.num_labels), target=labels.view(-1), reduction="none") 
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    model_test = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        num_labels=2,
)
    
    trainer_test = CustomTrainer(
        model=model_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )
    
#     # test
    test_data = pd.read_csv(BASEDATA_PATH + '/train_data.csv')[:20]
    # test_df['answer'] = 'A' # dummy answer that allows us to preprocess the test dataset just like we preprocessed the train dataset
    tokenized_test_dataset = Dataset.from_pandas(test_data).map(preprocess, remove_columns=['senquence', 'label'])
    print(tokenized_test_dataset)
    softmax = nn.Softmax(dim=1)
    Prediction_Output = trainer_test.predict(tokenized_test_dataset)
    print(torch.tensor(Prediction_Output.predictions), Prediction_Output.label_ids, Prediction_Output.metrics)
    # predictions_as_ids = np.argsort(-test_predictions, 1)
    # predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
    # predictions_as_string = test_df['prediction'] = [
    #     ' '.join(row) for row in predictions_as_answer_letters[:, :3]
    # ]
    # # # generate
    # submission = test_df[['id', 'prediction']]
    # submission.to_csv('submission.csv', index=False)

# pd.read_csv('submission.csv').head()