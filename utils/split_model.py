import os
import torch

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, top_k_top_p_filtering


MODEL_PATH = "/home/kaggleLLAM/model/llama2-7b"
SAVE_PATH = "/home/kaggleLLAM/model/llama2-7b-max5GB"


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_PATH,
#     use_fast=True,
#     trust_remote_code=True,
#     padding_side="right",
#     pad_token="<unk>",
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# exit()
model.save_pretrained(SAVE_PATH, max_shard_size="5GB")