# MEGA-DT - An RST Discourse Treebank with Structure and Nuclearity.
This is the official code for the paper ["MEGA RST Discourse Treebanks with Structure and Nuclearity from Scalable Distant Sentiment Supervision" (EMNLP 2020)](https://arxiv.org/abs/2011.03017), in which we present a novel scalable methodology to automatically generate discourse treebanks using distant supervision from sentiment-annotated datasets, creating MEGA-DT, a new large-scale discourse-annotated corpus.

Depending on your use-case, we offer four different entrance points to our work:
 1. Download the pre-trained discourse parsing model, trained using the [Two-Stage parser](https://github.com/yizhongw/StageDP) on MEGA-DT. This is recommended if you want to directly use our trained model as part of your system. To download the pre-trained model go to [our website](http://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/mega_dt.html) and request your version through the form.

 2. Download the MEGA-DT dataset, containing ~250,000 documents with full RST-style discourse trees (containing structure and nuclearity). This is recommended if you want to explore our dataset, train any available discourse parser or experiment with the data. To download the MEGA-DT dataset go to [our website](http://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/mega_dt.html) and request your version through the form.

 3. Download the pre-trained neural Multiple-Instance Learning model (adapted from [this](https://arxiv.org/abs/1711.09645) paper) to annotate any document with EDU-level sentiment and importance scores. This is recommended if you want to generate discourse trees according to the CKY-style approach yourself, or if you need access to the EDU-level sentiment data. To download the Multiple-Instance Learning model go to [our website](http://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/mega_dt.html) and request your version through the form.

 4. Start from the source-code. This allows you to train, run and evaluate our complete system from scratch including:
 	* Run preprocessing on the EDU-segmented Yelp'13 corpus
 	* Train, evaluate and use our MIL model (adapted from [this](https://arxiv.org/abs/1711.09645) paper) to obtain EDU-level sentiment and importance scores
 	* Run the CKY algorithm and generate the sentiment-guided MEGA-DT datset with structure and nuclearity attributes
 	* Train and evaluate the generated treebank using the Two-Stage parser (proposed in [this](https://www.aclweb.org/anthology/P17-2029/) paper) on RST-DT and other RST-style discourse datasets.

*If you decide to run our complete model, please follow these instructions:*

## Prerequisites
The presented code has been tested using `python 3`. Please ensure your python version is up-to-date.

To ensure the correct and compatible versions of all libraries, please install the specified versions from the requirements file:
```
pip install -r requirements.txt
```

Further, we use pre-trained GloVe embeddings, which need to be made accesible to the system. Per default, the 300-dimensional GloVe embeddings are expected under `./data/glove.txt`.

To run the discourse parser, CoreNLP is required. Please specify the location of the CoreNLP root folder (default: `./data/CoreNLP/stanford-corenlp-full-2018-10-05/`)

## Data
We include the EDU-segmented Yelp'13 train/dev/test datasets (as published in [this](https://arxiv.org/abs/1711.09645) paper) in the /data section, allowing you to directly train your model. We are not allowed to include the RST-DT and Instruction datasets.

## Execution
To run our model and generate RST-style discourse treebanks with structure and nuclearity from scalable distant sentiment supervision run the following command:
```
python main.py --complete
```

Please note that depending on the location of the train/dev/test data as well as further required files (GloVe embeddings, CoreNLP) the respective parameters need to be adjusted. To see the full set of availabe parameters, use
```
python main.py -h
```

## Cite this paper
Coming Soon.

