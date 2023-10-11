# LLM4Science

Welcome to the "LLM4Science" repository! This is a simple large language model for science exam, and it is mainly used for kaggle competition, and here is the [link](https://www.kaggle.com/competitions/kaggle-llm-science-exam/). Below is the more introduction.

## Main Method

We had won the Sliver medal of the contest, and finally reached Top 3% on the leardborad.

Our main method is using wiki pedia as the openbook, using the cleaned wiki knowdledge and using three our trained deberta model, finally using feature combining method.


## Background

Inspired by the [OpenBookQA dataset](https://allenai.org/data/open-book-qa), this competition challenges participants to answer difficult science-based questions *written by a Large Language Model*.

Your work will help researchers better understand the ability of LLMs to test themselves, and the potential of LLMs that can be run in resource-constrained environments.

The final score is based on the formula $$MAP@3=\frac1U\sum_{u=1}^U\sum_{k=1}^{min(n,3)}P(k)\times rel(k)$$, where $U$ is the number of questions in the test set, $P(k)$ is the precision at cutoff $k$, $n$ is the number predictions per question, and $rel(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label, zero otherwise.

Input File is like `id, prompt, A, B, C, D, E, answer` and submission file is like `id, prediction`. You may predict up to 3 labels for your `prediction`.

## File Structure

The repository's file structure is followed.

```
_________
	|____backup
		|____data
		|____code
	|____data
		|____270K_PKL
		|____270K-Wikipedia-STEM-articles
		|____wiki
		|____all_12_context.csv
		|____test_fixed.csv
		|____train.csv
		|____eval.csv
		|____train_context.csv
		|____eval_context.csv
	|____output
		|____llama2-7b-max5gb
		|____llama2-13b-max5gb
		|____llama2-7b-lora-wiki
		|____llama2-13b-lora-wiki
		|____deberta
			|____model_0914
			|____model_0920
			|____model_1002
			|____checkpoints_100
			|____checkpoints_103
			|____checkpoints_104
	|____utils
		|____cal_three_input.py
		|____cal_three_model.py
		|____shuffle_data.py
		|____split_model.py
		|____train_data_analyze.py
		|____add_E.py
		|____add_id.py
		|____extract_raw.py
		|____test_tokenizer.py
	|____scripts
		|____test_causal.py
		|____test_cls.py
		|____train_cls.py
		|____train_causal.py
		|____test_final.py
		|____train_causal_wiki.py
	|____add_wikipedia.py
	|____train_deberta.py
	|____test_deberta.py
	|____train_deberta_distillation.py
	|____train_deberta_270kcontext1.py
	|____train_deberta_270kcontext2.py
	|____requirements.txt
```

## Quick Start

- Install the requirements of this repository.
- Download the needed data.
- Change the code's path in the code.

```python
python add_wikipedia.py
python train_causal_wiki.py
python test_final.py
```

## TODO


## CITE

- [Dataset notebook](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383)

  [Infer notebook](https://www.kaggle.com/code/zzy990106/llama-7b-infer)

  [Model choose notebook](https://www.kaggle.com/code/radek1/best-open-source-llm-starter-pack)

  [Add wikipedia notebook](https://www.kaggle.com/code/jjinho/open-book-llm-science-exam)

  [270k cleaned wikipedia dataset](https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded)


- [LLM checkpoint 13b](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b)

  [LLM checkpoint 7b](https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b)

  [deberta]()

- kaggle competition

```
@misc{kaggle-llm-science-exam,
    author = {Will Lifferth, Walter Reade, Addison Howard},
    title = {Kaggle - LLM Science Exam},
    publisher = {Kaggle},
    year = {2023},
    url = {https://kaggle.com/competitions/kaggle-llm-science-exam}
}
```





