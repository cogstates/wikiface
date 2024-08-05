# Face and Politeness on Wikipedia

This is the repository corresponding to [Examining Gender and Power on Wikipedia Through Face and Politeness](https://arxiv.org/), which was accepted to [SIGDIAL 2024](https://2024.sigdial.org/). It contains the code and data used to train face act annotation models for Wikipedia talk pages. This includes the WikiFace Corpus, a subset of the [Wikipedia Talk Pages Corpus](https://convokit.cornell.edu/documentation/wiki.html) which we annotated for [face acts](https://en.wikipedia.org/wiki/Politeness_theory).

## Installation
Supposing [conda](https://docs.conda.io/en/latest/) and [poetry](https://python-poetry.org) are installed, the project dependencies can be setup using the following commands.

```
conda create -n wikiface python=3.10
conda activate wikiface
poetry install
```

By default, all scripts will log their output to `/home/{username}/scratch/logs/`. To change this behavior see ~line 40 of `src/core/context.py`.

## Content
A summary of the content and structure of the repository is shown below.

```
wikiface/
|- bin/
|  |- classification.py - trains face act classification models.
|- configs/
|  |- llama3.json       - configuration for training our reported model.
|  |- predict.json      - configuration for predicting with our reported model.
|- data/
|  |- wikiface/         - the wikiface corpus and all unannotated talk pages.
|- outputs/
|  |- ...               - default location (generated) for models and results.
|- src/
|  |- ...               - additional utilities.
```

## Example Usage

```
CUDA_VISIBLE_DEVICES=0 ./bin/classification.py configs/llama3.json
```

## Citation

```
@inproceedings{soubki-et-al-2024-examining,
    title = "Examining Gender and Power on Wikipedia Through Face and Politeness",
    author = "Soubki, Adil and Choi, Shyne and Rambow, Owen",
    booktitle = "25th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL 2024)",
    year = "2024",
    month="sept",
    publisher="Association for Computational Linguistics"
}
```

