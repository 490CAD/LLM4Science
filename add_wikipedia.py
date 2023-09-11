import os
import gc
import faiss
from tqdm.auto import tqdm
import pandas as pd
import blingfire as bf
import numpy as np
from faiss import write_index, read_index
from collections.abc import Iterable
from sentence_transformers import SentenceTransformer
import ctypes
import time
import torch
libc = ctypes.CDLL("libc.so.6")

# # load the DLL
# blf = None

# path = os.path.dirname(os.path.abspath(filename))
# # detect windows
# if platform.system() == "Windows":
#     blf = cdll.LoadLibrary(os.path.join(path, "blingfiretokdll.dll"))
# # detect Mac OSX
# elif platform.system() == "Darwin":
#     blf = cdll.LoadLibrary(os.path.join(path, "libblingfiretokdll.dylib"))
# else:
# # detect linux
#     blf = cdll.LoadLibrary(os.path.join(path, "libblingfiretokdll.so"))

BASEDATA_PATH = "/home/kaggleLLAM/data"

SIM_MODEL = "/home/kaggleLLAM/model/sentence-transformers_all-MiniLM-L6-v2"
MAX_LENGTH = 512
DEVICE = 0
BATCH_SIZE = 32
WIKI_PATH = "/home/kaggleLLAM/data/wikipedia"
wiki_files = os.listdir(WIKI_PATH)
## Parameter to determine how many relevant sentences to include
NUM_SENTENCES_INCLUDE = 30
NUM_TITLE_INCLUDE = 5

def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)

# title
print('-------------title--------------')
trn = pd.read_csv(BASEDATA_PATH + "/eval_66855.csv").drop(columns="id")
model = SentenceTransformer(SIM_MODEL, device='cuda')
model.max_seq_length = MAX_LENGTH
sentence_index = read_index(BASEDATA_PATH+"/wikipedia_202307.index")
prompt_embeddings = model.encode(trn.prompt.values, batch_size=BATCH_SIZE, device=0, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
_ = gc.collect()
print('-------------search-------------')
torch.cuda.empty_cache()
gpu_sentence_index = faiss.index_cpu_to_all_gpus(sentence_index)
time1 = time.time()
search_score, search_index = gpu_sentence_index.search(prompt_embeddings, NUM_TITLE_INCLUDE)
time2 = time.time()
print((time2 - time1))
del sentence_index
del gpu_sentence_index
del prompt_embeddings
_ = gc.collect()
libc.malloc_trim(0)
# sentence
print('-------------sentence-------------')
torch.cuda.empty_cache()
df = pd.read_parquet(WIKI_PATH + "/wiki_2023_index.parquet", columns=['id', 'file'])
wikipedia_file_data = []
for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
    scr_idx = idx
    _df = df.loc[scr_idx].copy()
    _df['prompt_id'] = i
    wikipedia_file_data.append(_df)
wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

del df
_ = gc.collect()
libc.malloc_trim(0)

wiki_text_data = []
for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
    _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].tolist()]
    _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

    _df_temp = _df[_df['id'].isin(_id)].copy()
    del _df
    _ = gc.collect()
    libc.malloc_trim(0)
    wiki_text_data.append(_df_temp)
wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
_ = gc.collect()

processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)
processed_wiki_text_data.to_csv("/home/kaggleLLAM/processed_wiki_text_data.csv")
wikipedia_file_data.to_csv("/home/kaggleLLAM/wikipedia_file_data.csv")
## Get embeddings of the wiki text data
print('-------------embeddings-------------')
torch.cuda.empty_cache()
print(processed_wiki_text_data.items())
wiki_data_embeddings = model.encode(processed_wiki_text_data.text,
                                    batch_size=BATCH_SIZE,
                                    device=DEVICE,
                                    show_progress_bar=True,
                                    convert_to_tensor=True,
                                    normalize_embeddings=True)#.half()
wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()
_ = gc.collect()

## Combine all answers
trn['answer_all'] = trn.apply(lambda x: " ".join([str(x['A']), str(x['B']), str(x['C']), str(x['D']), str(x['E'])]), axis=1)


## Search using the prompt and answers to guide the search
print('-------------pair-------------')
torch.cuda.empty_cache()
trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']

question_embeddings = model.encode(trn.prompt_answer_stem.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
question_embeddings = question_embeddings.detach().cpu().numpy()

## List containing just Context
contexts = []

for r in tqdm(trn.itertuples(), total=len(trn)):

    prompt_id = r.Index

    prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values

    if prompt_indices.shape[0] > 0:
        prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
        prompt_index.add(wiki_data_embeddings[prompt_indices])

        context = ""
        
        gpu_prompt_index = faiss.index_cpu_to_all_gpus(prompt_index)
        
        ## Get the top matches
        ss, ii = gpu_prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
        for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
            context += processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i] + " "
    contexts.append(context)
    _ = gc.collect()
_ = gc.collect()
trn['context'] = contexts
trn[["prompt", "context", "A", "B", "C", "D", "E", "answer"]].to_csv(BASEDATA_PATH + "/eval_context_2.csv", index=False)