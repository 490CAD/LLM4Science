from util_openbook import get_contexts, generate_openbook_output
import pickle

# get_contexts()
# generate_openbook_output()

import gc
gc.collect()

import numpy as np
import pandas as pd 
from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import LongformerTokenizer, LongformerForMultipleChoice
import transformers
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import unicodedata

import os

def SplitList(mylist, chunk_size):
    return [mylist[offs:offs+chunk_size] for offs in range(0, len(mylist), chunk_size)]

def get_relevant_documents_parsed(df_valid):
    df_chunk_size=600
    paraphs_parsed_dataset = load_from_disk("/home/kaggleLLAM/model/all-paraphs-parsed-expanded")
    modified_texts = paraphs_parsed_dataset.map(lambda example:
                                             {'temp_text':
                                              f"{example['title']} {example['section']} {example['text']}".replace('\n'," ").replace("'","")},
                                             num_proc=2)["temp_text"]
    
    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx: idx+df_chunk_size]
    
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)
        
    article_indices_array =  np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)
    
    top_per_query = article_indices_array.shape[1]
    articles_flatten = [(
                         articles_values_array[index],
                         paraphs_parsed_dataset[idx.item()]["title"],
                         paraphs_parsed_dataset[idx.item()]["text"],
                        )
                        for index,idx in enumerate(article_indices_array.reshape(-1))]
    retrieved_articles = SplitList(articles_flatten, top_per_query)
    return retrieved_articles



def get_relevant_documents(df_valid):
    df_chunk_size=800
    
    cohere_dataset_filtered = load_from_disk("/home/kaggleLLAM/model/stem-wiki-cohere-no-emb")
    modified_texts = cohere_dataset_filtered.map(lambda example:
                                             {'temp_text':
                                              unicodedata.normalize("NFKD", f"{example['title']} {example['text']}").replace('"',"")},
                                             num_proc=2)["temp_text"]
    
    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx: idx+df_chunk_size]
    
        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)
        
    article_indices_array =  np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)
    
    top_per_query = article_indices_array.shape[1]
    articles_flatten = [(
                         articles_values_array[index],
                         cohere_dataset_filtered[idx.item()]["title"],
                         unicodedata.normalize("NFKD", cohere_dataset_filtered[idx.item()]["text"]),
                        )
                        for index,idx in enumerate(article_indices_array.reshape(-1))]
    retrieved_articles = SplitList(articles_flatten, top_per_query)
    return retrieved_articles



def retrieval(df_valid, modified_texts):
    
    corpus_df_valid = df_valid.apply(lambda row:
                                     f'{row["prompt"]}\n{row["prompt"]}\n{row["prompt"]}\n{row["A"]}\n{row["B"]}\n{row["C"]}\n{row["D"]}\n{row["E"]}',
                                     axis=1).values
    vectorizer1 = TfidfVectorizer(ngram_range=(1,2),
                                 token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
                                 stop_words=stop_words)
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
                                 stop_words=stop_words,
                                 vocabulary=vocab_df_valid)
    vectorizer.fit(modified_texts[:500000])
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)
    
    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    chunk_size = 100000
    top_per_chunk = 10
    top_per_query = 10

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx: idx+chunk_size])
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)
    
    merged_top_scores = np.sort(top_values_array, axis=1)[:,-top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:,-top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]
    
    return articles_indices, merged_top_scores


def prepare_answering_input(
        tokenizer, 
        question,  
        options,   
        context,   
        model,
        max_seq_length=4096,
    ):
    c_plus_q   = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)
    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
    example_encoded = {
        "input_ids": input_ids.to(model.device.index),
        "attention_mask": attention_mask.to(model.device.index),
    }
    return example_encoded

stop_words = []
with open("/home/kaggleLLAM/data/stop_words.txt", "r") as file:
    for line in file:
        st = line.strip()
        if st:
            stop_words.append(st)

df_valid = pd.read_csv('/home/kaggleLLAM/data/fixed.csv')

trn = pd.read_csv("./test_context.csv")
with open("/home/kaggleLLAM/data/pkl/fixed_parsed.data", "rb") as file:
    retrieved_articles_parsed = pickle.load(file)
with open("/home/kaggleLLAM/data/pkl/fixed.data", "rb") as file:
    retrieved_articles = pickle.load(file)

from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice

models = [ "/home/kaggleLLAM/deberta_0914/model_v2", "/home/kaggleLLAM/checkpoints_100/checkpoint-471", "/home/kaggleLLAM/checkpoints_103/checkpoint-1169",
            "/home/kaggleLLAM/checkpoints_104/checkpoint-1752", "/home/kaggleLLAM/model_v2", "/home/kaggleLLAM/model_v3"]



def cal_two_best(model_dir1, model_dir2):
    # model_dir1 = "/home/kaggleLLAM/deberta_0914/model_v2"
    # model_dir2 = "/home/kaggleLLAM/checkpoints_104/checkpoint-1752"
    tokenizer = AutoTokenizer.from_pretrained(model_dir1)
    # tokenizer.truncation_side = 'left'
    model1 = AutoModelForMultipleChoice.from_pretrained(model_dir1).cuda()
    model2 = AutoModelForMultipleChoice.from_pretrained(model_dir2).cuda()


    predictions = []
    submit_ids = []

    temp_probability_1 = []
    temp_probability_2 = []

    len_context1 = 0
    number_context1 = 0
    len_context2 = 0
    number_context2 = 0
    max_context1 = 0
    max_context2 = 0

    for index in range(df_valid.shape[0]):
        columns = df_valid.iloc[index].values
        submit_ids.append(columns[0])
        question = columns[1]
        options = [columns[2], columns[3], columns[4], columns[5], columns[6]]
        # context1 = f"{retrieved_articles[index][-8][2]}\n{retrieved_articles[index][-7][2]}\n{retrieved_articles[index][-6][2]}\n{retrieved_articles[index][-5][2]}\n{retrieved_articles[index][-4][2]}\n{retrieved_articles[index][-3][2]}\n{retrieved_articles[index][-2][2]}\n{retrieved_articles[index][-1][2]}"
        context1 = f"{retrieved_articles[index][-4][2]}\n{retrieved_articles[index][-3][2]}\n{retrieved_articles[index][-2][2]}\n{retrieved_articles[index][-1][2]}"
        # context2 = f"{retrieved_articles[index][-6][2]}\n{retrieved_articles_parsed[index][-5][2]}\n{retrieved_articles_parsed[index][-4][2]}\n{retrieved_articles_parsed[index][-3][2]}\n{retrieved_articles_parsed[index][-2][2]}\n{retrieved_articles_parsed[index][-1][2]}"
        context2 = f"{retrieved_articles_parsed[index][-3][2]}\n{retrieved_articles_parsed[index][-2][2]}\n{retrieved_articles_parsed[index][-1][2]}"
        len_context1 += len(context1)
        max_context1 = max(max_context1, len(context1))
        number_context1 += 1
        len_context2 += len(context2)
        max_context2 = max(max_context2, len(context2))
        number_context2 += 1
        more_column = trn.iloc[index].values
        context3 = more_column[2]
        inputs1 = prepare_answering_input(
            tokenizer=tokenizer, question=question,
            options=options, context=context3[:5000],model=model1
            )
        inputs2 = prepare_answering_input(
            tokenizer=tokenizer, question=question,
            options=options, context=context2[:5000],model=model2
            )
        
        with torch.no_grad():
            outputs1 = model1(**inputs1)    
            losses1 = -outputs1.logits[0].detach().cpu().numpy()
            probability1 = torch.softmax(torch.tensor(-losses1), dim=-1)
            
        with torch.no_grad():
            outputs2 = model2(**inputs2)
            losses2 = -outputs2.logits[0].detach().cpu().numpy()
            probability2 = torch.softmax(torch.tensor(-losses2), dim=-1)
            
        probability_ = 0.3*probability1 + 0.7* probability2

        predict = np.array(list("ABCDE"))[np.argsort(probability_)][-3:].tolist()[::-1]

        temp_probability_1.append(probability1)
        temp_probability_2.append(probability2)
        predictions.append(predict)
        
    predictions = [" ".join(i) for i in predictions]

    pd.DataFrame({'id':submit_ids,'prediction':predictions}).to_csv('submission.csv', index=False)

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


    m = MAP_at_3(predictions, df_valid.answer.values)
    print(model_dir1, model_dir2)
    print("0.3 0.7 value is:", m)
    final_best = 0.0
    for i in range(100):
        cal_pre = []
        for pre in range(len(temp_probability_1)):
            probability_ = (i / 100) * temp_probability_1[pre] + (1 - i / 100) * temp_probability_2[pre]
            predict = np.array(list("ABCDE"))[np.argsort(probability_)][-3:].tolist()[::-1]
            cal_pre.append(predict)
        cal_pre = [" ".join(i) for i in cal_pre]
        tmp_m = MAP_at_3(cal_pre, df_valid.answer.values)
        if tmp_m > m:
            final_best = i
            m = tmp_m
    
    with open("/home/kaggleLLAM/utils/two.txt", "a") as file:
        file.write(model_dir1 + " " + model_dir2 + "\n")
        file.write("The best pro is: " + str(final_best / 100) + "\n")
        file.write("The best value is: " + str(m) + "\n")
        file.write("------------------\n")
    print("The best pro is:", final_best / 100)
    print("The best value is:", m)
    print('----------------------------')

for model_dir1 in models:
    for model_dir2 in models:
        print(model_dir1, model_dir2)
        cal_two_best(model_dir1, model_dir2)
        print('-------------------------------')
